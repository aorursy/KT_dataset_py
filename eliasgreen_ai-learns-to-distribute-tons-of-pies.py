# use this commands in 'Console' without '!' to install env

#!cd ../input/gymshops/gym-shops

#!pip install -e .
import gym

from gym import error, spaces, utils

from gym.utils import seeding



import itertools

import random

import time



class ShopsEnv(gym.Env):

  metadata = {'render.modes': ['human']}



  def __init__(self):

    self.state = [0, 0, 0]

    self.next_state = [0, 0, 0]

    self.done = False

    self.actions = list(itertools.permutations([1, 2, 3]))

    self.reward = 0

    self.time_tracker = 0

    

    self.remembered_states = []

    

    t = int( time.time() * 1000.0 )

    random.seed( ((t & 0xff000000) >> 24) +

                 ((t & 0x00ff0000) >>  8) +

                 ((t & 0x0000ff00) <<  8) +

                 ((t & 0x000000ff) << 24)   )

    

  def step(self, action_num):

    # check if the simulation is alredy done

    if self.done:

        return [self.state, self.reward, self.done, self.next_state]

    else:

        # select next state

        self.state = self.next_state

        

        # remember state

        self.remembered_states.append(self.state) 

    

        # increment time tracker

        self.time_tracker += 1

        

        # choose action according got action number

        action = self.actions[action_num]

        

        # update state using action (add pies)

        self.next_state = [x + y for x, y in zip(action, self.state)]

        

        # generate how much will be bought

        self.next_state[0] -= (3 + random.uniform(-0.1, 0.1))

        self.next_state[1] -= (1 + random.uniform(-0.1, 0.1))

        self.next_state[2] -= (2 + random.uniform(-0.1, 0.1))

        

        # select reward for action

        if any([x < 0 for x in self.next_state]):

            self.reward = sum([x for x in self.next_state if x < 0])

        else:

            self.reward = 1

            

        # reset shop if it has negative state

        if self.time_tracker >= 3:

            remembered_state = self.remembered_states.pop(0)

            self.next_state = [max(x - y, 0) for x, y in zip(self.next_state, remembered_state)]

        else:

            self.next_state = [max(x, 0) for x in self.next_state]

        

        

        # check if game is done

        self.done = self.time_tracker == 30



        return [self.state, self.reward, self.done, self.next_state]

    

  def reset(self):

    self.state = [0, 0, 0]

    self.next_state = [0, 0, 0]

    self.done = False

    self.reward = 0

    self.time_tracker = 0

    

    self.remembered_states = []

    

    t = int( time.time() * 1000.0 )

    random.seed( ((t & 0xff000000) >> 24) +

                 ((t & 0x00ff0000) >>  8) +

                 ((t & 0x0000ff00) <<  8) +

                 ((t & 0x000000ff) << 24)   )

                 

    return self.state

    

  def render(self, mode='human', close=False):

    print('-'*20)

    print('First shop')

    print('Pies:', self.state[0])



    print('Second shop')

    print('Pies:', self.state[1])



    print('Third shop')

    print('Pies:', self.state[2])

    print('-'*20)

    print('')
try:

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import gym # for environments

    import gym_shops # for our environment

    from tqdm import tqdm # progress tracker



    # for plots

    import matplotlib.pyplot as plt

    import seaborn as sns

    from IPython.display import clear_output

    sns.set_color_codes()



    # for modeling

    from collections import deque

    from keras.models import Sequential

    from keras.layers import Dense

    from keras.optimizers import Adam # adaptive momentum 

    import random # for stochasticity of our environment

except:

    pass
class DQLAgent(): 

    

    def __init__(self, env):

        # parameters and hyperparameters

        

        # this part is for neural network or build_model()

        self.state_size = 3 # this is for input of neural network node size

        self.action_size = 6 # this is for out of neural network node size

        

        # this part is for replay()

        self.gamma = 0.99

        self.learning_rate = 0.01

        

        # this part is for adaptiveEGreedy()

        self.epsilon = 0.99 # initial exploration rate

        self.epsilon_decay = 0.99

        self.epsilon_min = 0.0001

        

        self.memory = deque(maxlen = 5000) # a list with 5000 memory cell, if it becomes full first inputs will be deleted

        

        self.model = self.build_model()

    

    def build_model(self):

        # neural network for deep Q learning

        model = Sequential()

        model.add(Dense(10, input_dim = self.state_size, activation = 'sigmoid')) # first hidden layer

        model.add(Dense(50, activation = 'sigmoid')) # second hidden layer

        model.add(Dense(10, activation = 'sigmoid')) # third hidden layer

        model.add(Dense(self.action_size, activation = 'sigmoid')) # output layer

        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        return model

    

    def remember(self, state, action, reward, next_state, done):

        # storage

        self.memory.append((state, action, reward, next_state, done))

    

    def act(self, state):

        # acting, exploit or explore

        if random.uniform(0,1) <= self.epsilon:

            return random.choice(range(6))

        else:

            act_values = self.model.predict(state)

            return np.argmax(act_values[0])

            

    

    def replay(self, batch_size):

        # training

        

        if len(self.memory) < batch_size:

            return # memory is still not full

        

        minibatch = random.sample(self.memory, batch_size) # take batch_size random samples from memory

        for state, action, reward, next_state, done in minibatch:

            if done: # if the game is over, I dont have next state, I just have reward 

                target = reward

            else:

                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0]) 

                # target = R(s,a) + gamma * max Q`(s`,a`)

                # target (max Q` value) is output of Neural Network which takes s` as an input 

                # amax(): flatten the lists (make them 1 list) and take max value

            train_target = self.model.predict(state) # s --> NN --> Q(s,a) = train_target

            train_target[0][action] = target

            self.model.fit(state, train_target, verbose = 0) # verbose: dont show loss and epoch

    

    def adaptiveEGreedy(self):

        # decrease exploration rate

        

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    
try:

    # initialize gym environment and agent

    env = gym.make('shops-v0')

    agent = DQLAgent(env)



    # set training parameters

    batch_size = 100

    episodes = 1000



    # start training

    progress_bar = tqdm(range(episodes), position=0, leave=True)

    for e in progress_bar:

        # initialize environment

        state = env.reset()

        state = np.reshape(state, [1, 3])



        # track current time step, taken actions and sum of rewards for an episode

        time = 0

        taken_actions = []

        sum_rewards = 0





        # process episode

        while True:

            # act

            action = agent.act(state)



            # remember taken action

            taken_actions.append(action)



            # step

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, 3])



            # add got reward

            sum_rewards += reward



            # remember / storage

            agent.remember(state, action, reward, next_state, done)



            # update state

            state = next_state



            # replay

            agent.replay(batch_size)



            # adjust epsilon

            agent.adaptiveEGreedy()



            # increment time

            time += 1



            # show information about training state

            progress_bar.set_postfix_str(s='mean reward: {}, time: {}, epsilon: {}'.format(round(sum_rewards/time, 3), time, round(agent.epsilon, 3)), refresh=True)



            # check if the episode is already done

            if done:

                # show distribution of actions during current episode

                clear_output(wait=True)

                sns.distplot(taken_actions, color="y")

                plt.title('Episode: ' + str(e))

                plt.xlabel('Action number')

                plt.ylabel('Occurrence in %')

                plt.show()

                break

except:

    pass
try:

    import time

    trained_model = agent # now we have trained agent

    state = env.reset() # restart game

    state = np.reshape(state, [1,3])

    time_t = 0 # track time

    MAX_EPISOD_LENGTH = 1000

    taken_actions = []

    mean_reward = 0



    # simulate env with our trained model

    progress_bar = tqdm(range(MAX_EPISOD_LENGTH), position=0, leave=True)

    for time_t in progress_bar:

        # simulate one step

        action = trained_model.act(state)

        next_state, reward, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1,3])

        state = next_state

        taken_actions.append(action)



        # show result of the step



        clear_output(wait=True)

        env.render()

        progress_bar.set_postfix_str(s='time: {}'.format(time_t), refresh=True)

        print('Reward:', round(env.reward, 3))

        time.sleep(0.5)



        mean_reward += env.reward



        if done:

            break



    # show distribution of actions during current episode

    sns.distplot(taken_actions, color='y')

    plt.title('Test episode - mean reward: ' + str(round(mean_reward/(time_t+1), 3)))

    plt.xlabel('Action number')

    plt.ylabel('Occurrence in %')

    plt.show()    

except:

    pass