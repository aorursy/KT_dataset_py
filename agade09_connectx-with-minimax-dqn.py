import numpy as np

import gym

import os 

import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm

from kaggle_environments import evaluate, make
Train = True

gamma = 0.99

copy_step = 250

hidden_units = [100, 200, 200, 100]

max_experiences = 1000000

min_experiences = 100000

Steps_Till_Backprop = 64

batch_size = 512

lr = 1e-3

epsilon = 0.99

decay = 0.9999

min_epsilon = 0.1

episodes = 20000 #Set this to longer

precision = 5

Discard_Q_Value = -1e7

Metric_Titles = ['Max_Q','Avg_Q','Min_Q']

N_Downsampling_Episodes = 200
class ConnectX(gym.Env):

    def __init__(self):

        self.env = make('connectx', debug=False)



        # Define required gym fields (examples):

        config = self.env.configuration

        self.action_space = gym.spaces.Discrete(config.columns)

        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)



    def step(self, actions):

        return self.env.step(actions)



    def reset(self):

        return self.env.reset()

    

    def render(self, **kwargs):

        return self.env.render(**kwargs)



    def get_state(self):

        return self.env.state



    def game_over(self):

        return self.env.done



    def current_player(self):

        active = -1

        if self.env.state[0].status == "ACTIVE":

            active=0

        if self.env.state[1].status == "ACTIVE":

            active=1

        return active



    def get_configuration(self):

        return self.env.configuration
class DeepModel(tf.keras.Model):

    def __init__(self, num_states, hidden_units, num_actions):

        super(DeepModel, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))

        self.hidden_layers = []

        for i in hidden_units:

            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='relu', kernel_initializer='he_normal'))

        self.output_layer = tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer='RandomNormal')



    @tf.function

    def call(self, inputs):

        z = self.input_layer(inputs)

        for layer in self.hidden_layers:

            z = layer(z)

        output = self.output_layer(z)

        return output
class DQN:

    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.optimizer = tf.keras.optimizers.Nadam(lr)

        self.gamma = gamma

        self.model = DeepModel(num_states, hidden_units, num_actions)

        self.experience = {'inputs': [], 'a': [], 'r': [], 'inputs2': [], 'done': []} # The buffer

        self.max_experiences = max_experiences

        self.min_experiences = min_experiences



    def predict(self, inputs):

        return self.model(np.atleast_2d(inputs.astype('float32')))



    #@tf.function

    def train(self, TargetNet):

        # Only start the training process when we have enough experiences in the buffer

        if len(self.experience['inputs']) < self.min_experiences:

            return 0



        # Randomly select n experience in the buffer, n is batch-size

        ids = np.random.randint(low=0, high=len(self.experience['inputs']), size=self.batch_size)

        states = np.asarray([self.experience['inputs'][i] for i in ids])

        actions = np.asarray([self.experience['a'][i] for i in ids])

        rewards = np.asarray([self.experience['r'][i] for i in ids])



        # Prepare labels for training process

        states_next = np.asarray([self.experience['inputs2'][i] for i in ids])

        dones = np.asarray([self.experience['done'][i] for i in ids])

        

        # Find the value of the next states by computing the max over valid actions in these next states

        Move_Validity = states_next[:,:self.num_actions]==0

        Next_Q_Values = TargetNet.predict(states_next)

        Next_Q_Values = np.where(Move_Validity,Next_Q_Values,Discard_Q_Value)

        value_next = -np.max(Next_Q_Values,axis=1)

        

        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)



        with tf.GradientTape() as tape:

            selected_action_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)

            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))



    # Get an action by using epsilon-greedy

    def get_action(self, state, epsilon):

        prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()

        if np.random.random() < epsilon:

            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0])), prediction

        else:

            for i in range(self.num_actions):

                if state.board[i] != 0:

                    prediction[i] = Discard_Q_Value

            return int(np.argmax(prediction)) , prediction



    def add_experience(self, exp):

        if len(self.experience['inputs']) >= self.max_experiences:

            for key in self.experience.keys():

                self.experience[key].pop(0)

        for key, value in exp.items():

            self.experience[key].append(value)



    def copy_weights(self, TrainNet):

        variables1 = self.model.trainable_variables

        variables2 = TrainNet.model.trainable_variables

        for v1, v2 in zip(variables1, variables2):

            v1.assign(v2.numpy())



    def save_weights(self, path):

        self.model.save_weights(path)



    def load_weights(self, path):

        ref_model = tf.keras.Sequential()



        ref_model.add(self.model.input_layer)

        for layer in self.model.hidden_layers:

            ref_model.add(layer)

        ref_model.add(self.model.output_layer)



        ref_model.load_weights(path)

    

    # Each state is represented as 1s for the current player's pieces, -1s for the opponent's pieces and 0s

    def preprocess(self, state):

        return np.array([1 if val==state.mark else 0 if val==0 else -1 for val in state.board])
def play_game(env, TrainNet, TargetNet, epsilon, copy_step, Global_Step_Counter):

    turns = 0

    env.reset()

    Metric_Buffer={key:[] for key in Metric_Titles}

    while not env.game_over():

        active = env.current_player()



        # Using epsilon-greedy to get an action

        observations = env.get_state()[active].observation

        action, Q_Values = TrainNet.get_action(observations, epsilon)

        Q_Values = [val for val in Q_Values if val!=Discard_Q_Value]

        Metric_Buffer['Avg_Q'].append(np.mean(Q_Values))

        Metric_Buffer['Max_Q'].append(np.max(Q_Values))

        Metric_Buffer['Min_Q'].append(np.min(Q_Values))



        # Caching the information of current state

        prev_observations = observations



        # Take action

        env.step([action if i==active else None for i in [0,1]])



        reward=env.get_state()[active].reward



        #Convert environment's [0,0.5,1] reward scheme to [-1,1]

        if env.game_over():

            if reward == 1: # Won

                reward = 1

            elif reward == 0: # Lost

                reward = -1

            else: # Draw

                reward = 0

        else:

            reward = 0



        next_active = 1 if active==0 else 0

        # Adding experience into buffer

        observations = env.get_state()[next_active].observation

        exp = {'inputs': TrainNet.preprocess(prev_observations), 'a': action, 'r': reward, 'inputs2': TrainNet.preprocess(observations), 'done': env.game_over()}

        TrainNet.add_experience(exp)



        turns += 1

        total_turns = Global_Step_Counter+turns

        # Train the training model by using experiences in buffer and the target model

        if total_turns%Steps_Till_Backprop==0:

            TrainNet.train(TargetNet)

        if total_turns%copy_step==0:

            # Update the weights of the target model when reaching enough "copy step"

            TargetNet.copy_weights(TrainNet)

    results={key:[] for key in Metric_Titles}

    for metric_name in Metric_Titles:

        results[metric_name]=np.mean(Metric_Buffer[metric_name])

    return results, turns
env = ConnectX()



num_states = env.observation_space.n

num_actions = env.action_space.n



Metrics = {key:[] for key in Metric_Titles} #Here we will store metrics for plotting after training

Metrics_Buffer = {key:[] for key in Metric_Titles} #Downsampling buffer



# Initialize models

TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)



if Train:

    Global_Step_Counter=0

    pbar = tqdm(range(episodes))

    pbar2 = tqdm()

    for n in pbar:

        epsilon = max(min_epsilon, epsilon * decay)

        results, steps = play_game(env, TrainNet, TargetNet, epsilon, copy_step, Global_Step_Counter)

        Global_Step_Counter += steps

        for metric_name in Metric_Titles:

            Metrics_Buffer[metric_name].append(results[metric_name])



        if Global_Step_Counter%N_Downsampling_Episodes==0:

            for metric_name in Metric_Titles: #Downsample our metrics from the buffer

                Metrics[metric_name].append(np.mean(Metrics_Buffer[metric_name]))

                Metrics_Buffer[metric_name].clear()

            pbar.set_postfix({

                'Steps': Global_Step_Counter,

                'Updates': Global_Step_Counter*batch_size/Steps_Till_Backprop

            })



            pbar2.set_postfix({

                'max_Q': Metrics['Max_Q'][-1],

                'avg_Q': Metrics['Avg_Q'][-1],

                'min_Q': Metrics['Min_Q'][-1],

                'epsilon': epsilon,

                'turns': steps

            })



    def Plot(data,title):

        plt.figure()

        plt.plot(data)

        plt.xlabel('Episode')

        plt.ylabel(title)

        plt.savefig(title+'.png')

        plt.close()



    for metric_name in Metric_Titles:

        Plot(Metrics[metric_name],metric_name)



    TrainNet.save_weights('./weights.h5')

else:

    TrainNet.load_weights('./weights.h5')
fc_layers = []



# Get all hidden layers' weights

for i in range(len(hidden_units)):

    fc_layers.extend([

        TrainNet.model.hidden_layers[i].weights[0].numpy().tolist(), # weights

        TrainNet.model.hidden_layers[i].weights[1].numpy().tolist() # bias

    ])



# Get output layer's weights

fc_layers.extend([

    TrainNet.model.output_layer.weights[0].numpy().tolist(), # weights

    TrainNet.model.output_layer.weights[1].numpy().tolist() # bias

])



# Convert all layers into usable form before integrating to final agent

fc_layers = list(map(

    lambda x: str(list(np.round(x, precision))) \

        .replace('array(', '').replace(')', '') \

        .replace(' ', '') \

        .replace('\n', ''),

    fc_layers

))

fc_layers = np.reshape(fc_layers, (-1, 2))



# Create the agent

my_agent = '''def my_agent(observation, configuration):

    import numpy as np



'''



# Write hidden layers

for i, (w, b) in enumerate(fc_layers[:-1]):

    my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)

    my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)

# Write output layer

my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])

my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])



my_agent += '''

    board = observation.board[:]

    out = np.array([1 if val==observation.mark else 0 if val==0 else -1 for val in board],np.float32)

'''



# Calculate hidden layers

for i in range(len(fc_layers[:-1])):

    my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)

    my_agent += '    out = np.maximum(0,out)\n' # Relu function

# Calculate output layer

my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'

my_agent += '    out = np.tanh(out)\n'



my_agent += '''

    for i in range(configuration.columns):

        if observation.board[i] != 0:

            out[i] = -1e7



    return int(np.argmax(out))

    '''



with open('submission.py', 'w') as f:

    f.write(my_agent)
from submission import my_agent



def epsilon_greedify(agent,epsilon=0.05): #Greedify our agent so we don't play the same games over and over in strength evaluation

    def greedified_agent(observation,configuration):

        import random

        if random.random()<epsilon:

            return random.choice([i for i in range(num_actions) if observation.board[i]==0])

        else:

            return agent(observation,configuration)

    return greedified_agent



def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



my_agent = epsilon_greedify(my_agent)

print("My Agent vs. Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=20)))

print("Negamax Agent vs. My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=20)))