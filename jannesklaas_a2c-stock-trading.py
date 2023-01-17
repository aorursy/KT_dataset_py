# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/Data/Stocks"))

# Any results you write to the current directory are saved as output.
DATA_PATH = '../input/Data/Stocks'
class TradeEnv():    
    def reset(self):
        self.data = self.gen_universe()
        self.pos = 0
        self.game_length = self.data.shape[0]
        self.returns = []
        
        # return first state
        return self.data[0,:-1,:]
    
    def step(self,allocation):
        ret = np.sum(allocation * self.data[self.pos,-1,:])
        self.returns.append(ret)
        mean = 0
        std = 1
        if len(self.returns) >= 20:
            mean = np.mean(self.returns[-20:])
            std = np.std(self.returns[-20:]) + 0.0001
        sharpe = mean / std
        
        if (self.pos +1) >= self.game_length:
            return None, sharpe, True, {}  
        else:
            self.pos +=1
            return self.data[self.pos,:-1,:], sharpe, False, {}
        
    def gen_universe(self):
        stocks = os.listdir(DATA_PATH)
        stocks = np.random.permutation(stocks)
        frames = []
        idx = 0
        while len(frames) < 100:
            try:
                stock = stocks[idx]
                frame = pd.read_csv(os.path.join(DATA_PATH,stock),index_col='Date')
                frame = frame.loc['2005-01-01':].Close
                frames.append(frame)
            except: # catch *all* exceptions
                e = sys.exc_info()[0]
            idx += 1

        df = pd.concat(frames,axis=1,ignore_index=False)
        df = df.pct_change()
        df = df.fillna(0)
        batch = df.values
        episodes = []
        for i in range(batch.shape[0] - 101):
            eps = batch[i:i+101]
            episodes.append(eps)
        data = np.stack(episodes)
        assert len(data.shape) == 3
        assert data.shape[-1] == 100
        return data
class RandomTrader():
    def get_action(self):
        action = np.random.rand(100) * 2 - 1
        action = action * (np.abs(action) / np.sum(np.abs(action)))
        return action
import sys
#import gym
import numpy as np
from scipy.stats import norm
from keras.layers import Dense, Input, Lambda, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from collections import deque
import random

EPISODES = 3000


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, state_seq_length, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.state_size = state_size
        self.state_seq_length = state_seq_length
        self.action_size = action_size
        self.value_size = 1
        
        self.exp_replay = deque(maxlen=2000)

        # get gym environment name
        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        #self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        self.optimize_actor = self.actor_optimizer() #5
        self.optimize_critic = self.critic_optimizer() 


    def build_model(self):
        state = Input(batch_shape=(None, self.state_seq_length, self.state_size))
        
        x = LSTM(120,return_sequences=True)(state)
        x = LSTM(100)(x)
        
        actor_input = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
        # actor_hidden = Dense(self.hidden2, activation='relu')(actor_input)
        mu = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
        sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(30, activation='relu', kernel_initializer='he_uniform')(x)
        # value_hidden = Dense(self.hidden2, activation='relu')(critic_input)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_input)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        # mu = K.placeholder(shape=(None, self.action_size))
        # sigma_sq = K.placeholder(shape=(None, self.action_size))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_seq_length,self.state_size]))
        # sigma_sq = np.log(np.exp(sigma_sq + 1))
        epsilon = np.random.randn(self.action_size)
        # action = norm.rvs(loc=mu, scale=sigma_sq,size=1)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return action

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        self.exp_replay.append((state, action, reward, next_state, done))
        
        (state, action, reward, next_state, done) = random.sample(self.exp_replay,1)[0]
      
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.optimize_actor([state, action, advantages])
        self.optimize_critic([state, target])
state_size = 100
state_seq_length = 100
action_size = 100
import time
def run_experiment():
    start = time.time()
    env = TradeEnv()
    agent = A2CAgent(state_size, state_seq_length, action_size)
    epochs = 10
    reward_hist = []

    print('Setup: {:.4f}'.format(time.time() - start))

    for e in range(epochs):

        start = time.time()
        state = env.reset()
        state = np.reshape(state, [1,state_seq_length, state_size])
        done = False
        total_reward = 0
        print('Game Start: {:.4f}'.format(time.time() - start))

        while not done:

            start = time.time()
            action = agent.get_action(state)
            print('Get Action: {:.4f}'.format(time.time() - start))

            start = time.time()
            next_state, reward, done, info = env.step(action)
            print('Step: {:.4f}'.format(time.time() - start))

            start = time.time()
            next_state = np.reshape(next_state, [1,state_seq_length, state_size])
            agent.train_model(state, action, reward, next_state, done)
            print('Train: {:.4f}'.format(time.time() - start))

            total_reward += reward
            state = next_state

        print(total_reward)
        reward_hist.append(total_reward)
    return reward_hist
# Running training takes very long

#import matplotlib.pyplot as plt
#reward_hist = run_experiment()
#plt.plot(reward_hist)

