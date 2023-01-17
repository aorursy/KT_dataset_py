# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import inspect

import random

import gym

import tensorflow as tf

import tensorflow.keras.layers as kl

import tensorflow.keras.losses as kls

import tensorflow.keras.optimizers as ko

import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

from collections import deque

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)

env.render()
env.agents
env.configuration
env.specification
def my_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
# Play as first position against random agent.

trainer = env.train([None, "random"])

observation = trainer.reset()



print("Observation contains:\t", observation)

print("Configuration contains:\t", env.configuration)
my_action = my_agent(observation, env.configuration)

print("My Action", my_action)

observation, reward, done, info = trainer.step(my_action)

env.render(mode="ipython", width=100, height=90, header=False, controls=False)

print("Observation after:\t", observation)
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    print(reward)

env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    return sum(r[0] for r in rewards)/sum(r[0]+r[1] for r in rewards)

# Run multiple episodes to estimate its performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=100)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=100)))
# Negamax algorithm

print(inspect.getsource(env.agents['negamax']))
# random agent algorithm

print(inspect.getsource(env.agents['random']))
class ConnectX(gym.Env):

    def __init__(self):

        self.env = make("connectx", debug=True)

        self.pair = [None,"negamax"]

        self.config = self.env.configuration

        self.trainer = self.env.train(self.pair)

        

        # Define required gym fields (examples):

        config = self.env.configuration

        self.action_space = gym.spaces.Discrete(config.columns)

        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

        

    def step(self,action):

        return self.trainer.step(action)

    def reset(self):

        return self.trainer.reset()

    def render(self, **kwargs):

        return self.env.render(**kwargs)
class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits,  **kwargs):

        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
class Model(tf.keras.Model):

    def __init__(self, env, num_actions):

        super(Model, self).__init__('mlp_policy')

        self.env = env

        self.num_actions = num_actions

        self.hidden1 = kl.Dense(128, activation='relu')

        self.hidden2 = kl.Dense(128, activation='relu')

        self.value = kl.Dense(1, name='value')

        # Logits are unnormalized log probabilities.

        self.logits = kl.Dense(num_actions,  name='policy_logits')

        self.dist = ProbabilityDistribution()

        self.action_ = None

        self.value_ = None

        self.space = None

        self.empty = []

        

    def call(self, inputs, **kwargs):

        # Inputs is a numpy array, convert to a tensor.

        x = tf.convert_to_tensor(inputs)

        # Separate hidden layers from the same input tensor.

        hidden_logs = self.hidden1(x)

        hidden_vals = self.hidden2(x)

        return self.logits(hidden_logs), self.value(hidden_vals)

    

    def action_value(self, obs):

        # Executes `call()` under the hood.

        logits, values = self.predict_on_batch(obs)

        action = self.dist.predict_on_batch(logits)

        # Another way to sample actions:

        #   action = tf.random.categorical(logits, 1)

        # Will become clearer later why we don't use it.

        

        # The recursion shown below works absolutely fine but, 

        # while commit I am facing recursion error, so I have commented this out

        # This recursion prevents invalid column problem.

        

        #self.action_,  self.value_ = np.squeeze(action, axis = -1), np.squeeze(values, axis=-1)

        #self.space = [c for c in range(self.env.config.columns) if (obs[0][c] == 0)]

        #if self.action_ not in self.space and self.space!=self.empty:

        #    self.action_value(obs)

        return np.squeeze(action, axis = -1), np.squeeze(values, axis=-1)

    

    def preprocess(self, state):

        result = state.board[:]

        result.append(state.mark)



        return result
env = ConnectX()

model = Model(env, num_actions=env.action_space.n)

obs = env.reset()

obs = np.array(model.preprocess(obs))

# No feed_dict or tf.Session() needed at all!

action, value = model.action_value(obs[None, :])

print("Action: " +str(action)+", Value: " + str(value))
K.clear_session()

class Agent_Advanced:

    def __init__(self, model, lr=7e-3, gamma=0.8, value_c=0.5, entropy_c=1e-4):

        # Coefficients are used for the loss terms.

        self.value_c = value_c

        self.entropy_c = entropy_c

        # `gamma` is the discount factor

        self.gamma = gamma

        self.model = model

        self.model.compile(

                          optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),

                          # Define separate losses for policy logits and value estimate.

                          loss=[self._logits_loss, self._value_loss]

        )

    def train(self, env, batch_sz=64, updates=500):

        

        # Training loop: collect samples, send to optimizer, repeat updates times.

        ep_rewards = [0.0]

        next_obs = env.reset()

        next_obs = np.array(model.preprocess(next_obs))

        # Storage helpers for a single batch of data.

        actions = np.empty((batch_sz,), dtype=np.int32)

        rewards, dones, values = np.zeros((3, batch_sz,))

        

        observations = np.empty((batch_sz,len(next_obs.copy())) + env.observation_space.shape)

        for update in range(updates):

            for step in range(batch_sz):

                observations[step] = next_obs.copy()

                #print(observations[step])

                actions[step], values[step] = self.model.action_value(next_obs[None, :])

                next_obs, rewards[step], dones[step], _ = env.step(int(actions[step]))

                

                #values[step] = np.where(dones[step], rewards[step], rewards[step]+self.gamma*values[step])

                

                if rewards[step] >= 0.5: # Won

                    rewards[step] = 20

                elif rewards[step] == 0.0: # Lost

                    rewards[step] = -20

                else: # Draw

                    rewards[step] = 0.05

                ep_rewards[-1] += rewards[step]    

                

                next_obs = np.array(model.preprocess(next_obs))    

                #print(rewards[step])

                if dones[step]:

                    ep_rewards.append(0.0)

                    next_obs = env.reset()

                    next_obs = np.array(model.preprocess(next_obs))

                    print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

                    

                    

                

            _, next_value = self.model.action_value(next_obs[None, :])

            #next_value= np.where(dones, rewards, rewards+self.gamma*values)

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # To input actions and advantages through same API.

            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch

            losses = self.model.fit(observations, [acts_and_advs, returns])

            

            print("[%d/%d] Losses: %s" % (update + 1, updates, losses.history['loss']))



        return ep_rewards

    

    def _returns_advantages(self, rewards, done, values, next_value):

        # `next_value` is the bootstrap value estimate of the future state (critic).

        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        

        # Returns are calculated as discounted sum of future rewards.

        for t in reversed(range(rewards.shape[0])):

            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1 - done[t])

        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).

        advantages = returns - values

        

        return returns, advantages

        

    def _value_loss(self, return_, value):

        # Value loss is typically MSE between value estimates and returns.

        return self.value_c * kls.mean_squared_error(return_, value)

    

    def _logits_loss(self, actions_and_advantages, logits):

        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.

        # `from_logits` argument ensures transformation into normalized probabilities.

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.

        # Note: we only calculate the loss on the actions we've actually taken.

        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        

        # Entropy loss can be calculated as cross-entropy over itself.

        probs = tf.nn.softmax(logits)

        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.

        # Here signs are flipped because the optimizer minimizes.

        return policy_loss - self.entropy_c * entropy_loss
env = ConnectX()

model = Model(env, num_actions=env.action_space.n)

model.run_eagerly = True

print("Eager Execution:  ", tf.executing_eagerly())

print("Eager Keras Model:", model.run_eagerly)
agent = Agent_Advanced(model)

rewards_history = agent.train(env)

print("Finished training, testing....")
plt.figure(figsize=[20,10])

plt.plot(rewards_history)

plt.xlabel('Episode')

plt.ylabel('Avg rewards ')

plt.show()
K.clear_session()

class Agent_deepQ:

    def __init__(self, enviroment, optimizer):

        

        # Initialize atributes

        self.environment = environment

        self._state_size = enviroment.observation_space.n

        self._action_size = enviroment.action_space.n

        self._optimizer = optimizer

        self.space = None

        self.empty = []

        self.action_ = None



        self.expirience_replay = deque(maxlen=2000)



        # Initialize discount and exploration rate

        self.gamma = 0.7

        self.epsilon = 0.1



        # Build networks

        self.q_network = self._build_compile_model()

        self.target_network = self._build_compile_model()

        self.alighn_target_model()

        

    def _build_compile_model(self):

        model = tf.keras.Sequential()

        model.add(kl.Embedding(self._state_size, 100, input_length=1))

        model.add(kl.Reshape((100,)))

        model.add(kl.Dense(256, activation='relu'))

        model.add(kl.Dense(128, activation='relu'))

        model.add(kl.Dense(self._action_size, activation='linear'))

        

        model.compile(loss='mse', optimizer=self._optimizer)

        return model

    def alighn_target_model(self):

        self.target_network.set_weights(self.q_network.get_weights())

        

    def store(self, state, action, reward, next_state, terminated):

        self.expirience_replay.append((state, action, reward, next_state, terminated))

    

    def act(self, state):

        if np.random.rand() <= self.epsilon:

            self.action_ = int(np.random.choice([c for c in range(environment.config.columns) if state[c] == 0]))

        

        else:

            q_values = self.q_network.predict(state)        

            self.action_ = int(np.argmax(q_values[0]))

        self.space = [c for c in range(self.environment.config.columns) if (state[c] == 0)]

        if self.action_ not in self.space and self.space!=self.empty:

            self.act(state)

        return self.action_

    

    def preprocess(self, state):

        result = state.board[:]

        result.append(state.mark)

        return result

    

    def train(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state)

            if terminated:

                target[0][action] = reward

            else:

                next_state = self.preprocess(next_state)

                t = self.target_network.predict(next_state)

                target[0][action] = reward + self.gamma*np.amax(t)

                self.q_network.fit(np.array(state), np.array(target), epochs=1, verbose=0)

            
environment = ConnectX()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

agent = Agent_deepQ(environment, optimizer)



batch_size = 32

num_of_episodes = 800

timesteps_per_episode = 500

agent.q_network.summary()
reward_ = 0

total_reward = []

for e in range(num_of_episodes):

    # Reset the enviroment

    state = environment.reset()    

    terminated=False

    for time_step in range(timesteps_per_episode):

        state = agent.preprocess(state)

        # Run Action

        action = agent.act(state)

        # Take action 

        next_state, reward, terminated, info = environment.step(action)

        agent.store(state, action, reward, next_state, terminated)

        state = next_state

        reward_+=reward

        if terminated:

            agent.alighn_target_model()

            total_reward.append(reward_)

            reward_ = 0

            break

            

        if len(agent.expirience_replay) > batch_size:

            agent.train(batch_size)

    if (e + 1) % 10 == 0:

        print("**********************************")

        print("Episode: {}".format(e + 1))

        environment.render()

        print("**********************************")
plt.figure(figsize=[20,10])

plt.plot(total_reward)

plt.xlabel('Episode')

plt.ylabel('Avg rewards')

plt.show()