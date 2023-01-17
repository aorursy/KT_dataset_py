import numpy as np

class SumTree(object):

    data_pointer = 0

    

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0

    def __init__(self, capacity):

        # Number of leaf nodes (final nodes) that contains experiences

        self.capacity = capacity

        

        # Generate the tree with all nodes values = 0

        # To understand this calculation (2 * capacity - 1) look at the schema below

        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)

        # Parent nodes = capacity - 1

        # Leaf nodes = capacity

        self.tree = np.zeros(2 * capacity - 1)

        

        # Contains the experiences (so the size of data is capacity)

        self.data = np.zeros(capacity, dtype=object)

    

    

    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:

    def add(self, priority, data):

        # Look at what index we want to put the experience

        tree_index = self.data_pointer + self.capacity - 1



        # Update data frame

        self.data[self.data_pointer] = data



        # Update the leaf

        self.update (tree_index, priority)



        # Add 1 to data_pointer

        self.data_pointer += 1



        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)

            self.data_pointer = 0

            

    # Update the leaf priority score and propagate the change through tree

    def update(self, tree_index, priority):

        # Change = new priority score - former priority score

        change = priority - self.tree[tree_index]

        self.tree[tree_index] = priority



        # then propagate the change through tree

        # this method is faster than the recursive loop in the reference code

        while tree_index != 0:

            tree_index = (tree_index - 1) // 2

            self.tree[tree_index] += change

        

    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index, priority value of that leaf and experience associated with that leaf index:

    def get_leaf(self, v):

        parent_index = 0



        # the while loop is faster than the method in the reference code

        while True:

            left_child_index = 2 * parent_index + 1

            right_child_index = left_child_index + 1



            # If we reach bottom, end the search

            if left_child_index >= len(self.tree):

                leaf_index = parent_index

                break

            else: # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:

                    parent_index = left_child_index

                else:

                    v -= self.tree[left_child_index]

                    parent_index = right_child_index



        data_index = leaf_index - self.capacity + 1



        return leaf_index, self.tree[leaf_index], self.data[data_index]

    

    @property

    def total_priority(self):

        return self.tree[0] # Returns the root node



# Now we finished constructing our SumTree object, next we'll build a memory object.

class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree

    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken

    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly

    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    

    PER_b_increment_per_sampling = 0.001

    

    absolute_error_upper = 1.  # clipped abs error



    def __init__(self, capacity):

        # Making the tree 

        self.tree = SumTree(capacity)

        

    # Next, we define a function to store a new experience in our tree.

    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).

    def store(self, experience):

        # Find the max priority

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])



        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected

        # So we use a minimum priority

        if max_priority == 0:

            max_priority = self.absolute_error_upper



        self.tree.add(max_priority, experience)   # set the max priority for new priority

        

    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.

    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.

    # - Then a value is uniformly sampled from each range.

    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.

    def sample(self, n):

        # Create a minibatch array that will contains the minibatch

        minibatch = []



        b_idx = np.empty((n,), dtype=np.int32)



        # Calculate the priority segment

        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges

        priority_segment = self.tree.total_priority / n       # priority segment



        for i in range(n):

            # A value is uniformly sample from each range

            a, b = priority_segment * i, priority_segment * (i + 1)

            value = np.random.uniform(a, b)



            # Experience that correspond to each value is retrieved

            index, priority, data = self.tree.get_leaf(value)



            b_idx[i]= index



            minibatch.append([data[0],data[1],data[2],data[3],data[4]])



        return b_idx, minibatch

    

    # Update the priorities on the tree

    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.PER_e  # convert to abs and avoid 0

        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)

        ps = np.power(clipped_errors, self.PER_a)



        for ti, p in zip(tree_idx, ps):

            self.tree.update(ti, p)
import os, time, random, gym, pylab

import numpy as np

from collections import deque

from keras.models import Model, load_model

from keras.layers import Input, Dense, Lambda, Add

from keras.optimizers import Adam, RMSprop

from keras import backend as K

# from PER import *





def OurModel(input_shape, action_space, dueling):

    X_input = Input(input_shape)

    X = X_input



    # 'Dense' is the basic form of a neural network layer

    # Input Layer of state size(4) and Hidden Layer with 512 nodes

    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)



    # Hidden layer with 256 nodes

    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    

    # Hidden layer with 64 nodes

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)



    if dueling:

        state_value = Dense(1, kernel_initializer='he_uniform')(X)

        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)



        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)

        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)



        X = Add()([state_value, action_advantage])

    else:

        # Output Layer with # of actions: 2 nodes (left, right)

        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)



    model = Model(inputs = X_input, outputs = X, name='CartPole_D3QN_model')

    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])



    model.summary()

    return model



class DQNAgent:

    def __init__(self, env_name):

        self.env_name = env_name       

        self.env = gym.make(env_name)

        self.env.seed(0)  

        # by default, CartPole-v1 has max episode steps = 500

        self.env._max_episode_steps = 4000

        self.state_size = self.env.observation_space.shape[0]

        self.action_size = self.env.action_space.n



        self.EPISODES = 100

        memory_size = 10000

        self.MEMORY = Memory(memory_size)

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate



        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy

        self.epsilon = 1.0 # exploration probability at start

        self.epsilon_min = 0.01 # minimum exploration probability

        self.epsilon_decay = 0.0005 # exponential decay rate for exploration prob

        

        self.batch_size = 32



        # defining model parameters

        self.ddqn = True # use doudle deep q network

        self.Soft_Update = False # use soft parameter update

        self.dueling = True # use dealing netowrk

        self.epsilot_greedy = False # use epsilon greedy strategy

        self.USE_PER = True



        self.TAU = 0.1 # target network soft update hyperparameter



        self.scores, self.episodes, self.average = [], [], []



        self.Model_name = self.env_name+"_e_greedy.h5"

        

        # create main model and target model

        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size, dueling = self.dueling)

        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size, dueling = self.dueling)



    # after some time interval update the target model to be same with model

    def update_target_model(self):

        if not self.Soft_Update and self.ddqn:

            self.target_model.set_weights(self.model.get_weights())

            return

        if self.Soft_Update and self.ddqn:

            q_model_theta = self.model.get_weights()

            target_model_theta = self.target_model.get_weights()

            counter = 0

            for q_weight, target_weight in zip(q_model_theta, target_model_theta):

                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU

                target_model_theta[counter] = target_weight

                counter += 1

            self.target_model.set_weights(target_model_theta)



    def remember(self, state, action, reward, next_state, done):

        experience = state, action, reward, next_state, done

        if self.USE_PER:

            self.MEMORY.store(experience)

        else:

            self.memory.append((experience))



    def act(self, state, decay_step):

        # EPSILON GREEDY STRATEGY

        if self.epsilot_greedy:

        # Here we'll use an improved version of our epsilon greedy strategy for Q-learning

            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)

        # OLD EPSILON STRATEGY

        else:

            if self.epsilon > self.epsilon_min:

                self.epsilon *= (1-self.epsilon_decay)

            explore_probability = self.epsilon

    

        if explore_probability > np.random.rand():

            # Make a random action (exploration)

            return random.randrange(self.action_size), explore_probability

        else:

            # Get action from Q-network (exploitation)

            # Estimate the Qs values state

            # Take the biggest Q value (= the best action)

            return np.argmax(self.model.predict(state)), explore_probability



    def replay(self):

        if self.USE_PER:

            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)

        else:

            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))



        state = np.zeros((self.batch_size, self.state_size))

        next_state = np.zeros((self.batch_size, self.state_size))

        action, reward, done = [], [], []



        # do this before prediction

        # for speedup, this could be done on the tensor level

        # but easier to understand using a loop

        for i in range(self.batch_size):

            state[i] = minibatch[i][0]

            action.append(minibatch[i][1])

            reward.append(minibatch[i][2])

            next_state[i] = minibatch[i][3]

            done.append(minibatch[i][4])



        # do batch prediction to save speed

        # predict Q-values for starting state using the main network

        target = self.model.predict(state)

        target_old = np.array(target)

        # predict best action in ending state using the main network

        target_next = self.model.predict(next_state)

        # predict Q-values for ending state using the target network

        target_val = self.target_model.predict(next_state)



        for i in range(len(minibatch)):

            # correction on the Q value for the action used

            if done[i]:

                target[i][action[i]] = reward[i]

            else:

                if self.ddqn: # Double - DQN

                    # current Q Network selects the action

                    # a'_max = argmax_a' Q(s', a')

                    a = np.argmax(target_next[i])

                    # target Q Network evaluates the action

                    # Q_max = Q_target(s', a'_max)

                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   

                else: # Standard - DQN

                    # DQN chooses the max Q value among next actions

                    # selection and evaluation of action is on the target Q Network

                    # Q_max = max_a' Q_target(s', a')

                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))



        if self.USE_PER:

            indices = np.arange(self.batch_size, dtype=np.int32)

            absolute_errors = np.abs(target_old[indices, np.array(action)]-target[indices, np.array(action)])

            # Update priority

            self.MEMORY.batch_update(tree_idx, absolute_errors)



        # Train the Neural Network with batches

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)



    def load(self, name):

        self.model = load_model(name)



    def save(self, name):

        self.model.save(name)



    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):

        self.scores.append(score)

        self.episodes.append(episode)

        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        pylab.plot(self.episodes, self.average, 'r')

        pylab.plot(self.episodes, self.scores, 'b')

        pylab.ylabel('Score', fontsize=18)

        pylab.xlabel('Steps', fontsize=18)

        dqn = 'DQN_'

        softupdate = ''

        dueling = ''

        greedy = ''

        PER = ''

        if self.ddqn: dqn = 'DDQN_'

        if self.Soft_Update: softupdate = '_soft'

        if self.dueling: dueling = '_Dueling'

        if self.epsilot_greedy: greedy = '_Greedy'

        if self.USE_PER: PER = '_PER'

        try:

            pylab.savefig(dqn+self.env_name+softupdate+dueling+greedy+PER+".png")

        except OSError:

            pass



        return str(self.average[-1])[:5]

    

    def run(self):

        decay_step = 0

        for e in range(self.EPISODES):

            if e >= (self.EPISODES-1):

                self.save(self.Model_name)

            state = self.env.reset()

            start_time = time.time()

            state = np.reshape(state, [1, self.state_size])

            done = False

            i = 0

            while not done:

                #self.env.render()

                decay_step += 1

                action, explore_probability = self.act(state, decay_step)

                next_state, reward, done, _ = self.env.step(action)

                next_state = np.reshape(next_state, [1, self.state_size])

                if not done or i == self.env._max_episode_steps-1:

                    reward = reward

                else:

                    reward = -100

                self.remember(state, action, reward, next_state, done)

                state = next_state

                i += 1

                if done:

                    # every step update target model

                    self.update_target_model()

                    

                    # every episode, plot the result

                    average = self.PlotModel(i, e)

                     

                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, explore_probability, average))

                    end_time = time.time()

                    total = end_time - start_time

                    h = total//3600

                    m = (total%3600)//60

                    s = total%60

                    print("Total time spent: %i hours, %i minutes, and %i seconds" %(h, m, s))

                    if i == self.env._max_episode_steps:

                        print("Saving trained model to", self.Model_name)

#                         self.save(self.Model_name)

                        break

                self.replay()

        self.env.close()



    def test(self):

        self.load(self.Model_name)

        for e in range(self.EPISODES):

            state = self.env.reset()

            state = np.reshape(state, [1, self.state_size])

            done = False

            i = 0

            while not done:

                self.env.render()

                action = np.argmax(self.model.predict(state))

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(next_state, [1, self.state_size])

                i += 1

                if done:

                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))

                    break



if __name__ == "__main__":

    env_name = 'CartPole-v1'

    agent = DQNAgent(env_name)

    agent.run()

    #agent.test()
