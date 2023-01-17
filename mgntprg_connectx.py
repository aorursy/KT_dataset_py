!pip install 'kaggle-environments>=0.1.6'
import tensorflow as tf

import numpy as np
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
# This agent random chooses a non-empty column.

def my_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
env.reset()

# Play as the first agent against default "random" agent.

env.run([my_agent, "random"])

env.render(mode="ipython", width=500, height=450)
trainer = env.train([None, "negamax"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print(my_action)

    print(env.configuration)

    print(observation)

    observation, reward, done, info = trainer.step(my_action)

    break
from collections import deque

import random



DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 50_000

MIN_REPLAY_MEMORY_SIZE = 1_000

MINIBATCH_SIZE = 32

UPDATE_TARGET_EVERY = 5





class DQNAgent():

    def __init__(self):

        self.model = self.create_model()

        

        self.target_model = self.create_model()

        self.target_model.set_weights(self.model.get_weights())

        

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        

        self.target_update_counter = 0 # When it reaches a threshold, update target network weights

        

    def create_model(self):

        rows = env.configuration['rows']

        cols = env.configuration['columns']

        

        model = tf.keras.Sequential()

        

        model.add(tf.keras.layers.Conv2D(128, (2, 2), input_shape=(rows, cols, 1), activation='relu', padding='same'))

        model.add(tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same'))

        #model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(256, activation='relu'))

        model.add(tf.keras.layers.Dense(cols)) # Number of columns corresponds to action space size

        

        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        return model

    

    def update_replay_memory(self, transition):

        self.replay_memory.append(transition)

    

    def train(self, terminal_state, step):

        if(len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE):  # train only when you have samples

            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) # get random samples to train on 

        

        # predict q values for these states

        current_states = np.array([transition[0] for transition in minibatch])

        

        current_qs_list = self.model.predict(current_states)

        

        # get future states, query for NN

        new_current_states = np.array([transition[3] for transition in minibatch])

        future_qs_list = self.target_model.predict(new_current_states)

        

        X = []

        y = []

        

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:

                max_future_q = np.max(future_qs_list[index]) # For the example, find the best result as predicted by target_model

                new_q = reward + DISCOUNT * max_future_q

            else:

                new_q = reward

                

            current_qs = current_qs_list[index]

            current_qs[action] = new_q



            X.append(current_state)

            y.append(current_qs)

        

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        

        if terminal_state:

            self.target_update_counter += 1

        

        if self.target_update_counter > UPDATE_TARGET_EVERY:

            self.target_model.set_weights(self.model.get_weights())

            self.target_update_counter = 0

                

    def get_qs(self, state):

        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
agent = DQNAgent()

agent.get_qs(np.random.rand(6, 7, 1))
agent.model.summary()
from tqdm import tqdm

EPISODES = 2000

epsilon = 1  # not a constant, going to be decayed

EPSILON_DECAY = 0.99975

MIN_EPSILON = 0.001

from time import sleep



for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):

    episode_reward = 0

    step = 1

    

    current_state = env.reset()

    

    done = False

    while not done:

        # check if action is valid before performing it

        valid = False

        

        while not valid:

            try:

                board_obs = np.array(current_state[0]['observation']['board']).reshape((env.configuration['rows'], env.configuration['columns'], 1))

            except:

                board_obs = np.array(current_state['board']).reshape((env.configuration['rows'], env.configuration['columns'], 1))

            

            if np.random.random() > epsilon:

                action = np.argmax(agent.get_qs(board_obs))

            else:

                action = np.random.randint(0, env.configuration['columns'])

                

            for r in range(env.configuration['rows']):

                if board_obs[r, action, 0] == 0: # if 0, the action is valid

                    valid = True

        

        new_observation, reward, done, info = trainer.step(int(action))

        try:

            new_board_obs = np.array(new_observation[0]['board']).reshape((env.configuration['rows'], env.configuration['columns'], 1))

        except:

            new_board_obs = np.array(new_observation['board']).reshape((env.configuration['rows'], env.configuration['columns'], 1))

        

        try:

            episode_reward += reward

        except:

            #print(new_observation)

            print('Failure:', board_obs[:, :, 0])

        

        #agent.update_replay_memory((board_obs, action, reward, new_observation, done))

        agent.update_replay_memory((board_obs, action, reward, new_board_obs, done))

        agent.train(done, step)

        current_state = new_observation

        step += 1

        

        if epsilon > MIN_EPSILON:

            epsilon *= EPSILON_DECAY

            epsilon = max(MIN_EPSILON, epsilon)
arr = np.random.rand(4, 6, 7, 1)

print(agent.target_model.predict(arr))
def my_dqn_agent(observation, configuration):

    obs = np.array(observation.board).reshape(configuration.rows, configuration.columns, 1)

    

    valid = False

    action = None

    

    actions = agent.get_qs(obs)

    while not valid:

        action = np.argmax(actions)



        for r in range(env.configuration['rows']):

            if obs[r, action, 0] == 0: # if 0, the action is valid

                valid = True

            if not valid:

                actions[action] = -30

    

    #return int(np.argmax(agent.get_qs(obs)))

    return int(action)
env.run([my_dqn_agent, "random"])

env.render(mode="ipython", width=500, height=450)