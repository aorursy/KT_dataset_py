!pip install keras-rl
!pip install h5py

!pip install gym
!pip install gym[atari]
import gym

from keras.models import Sequential

import numpy as np

from keras.layers import Flatten,Dense,Activation

from keras.optimizers import Adam



from rl.policy import EpsGreedyQPolicy

from rl.memory import SequentialMemory

from rl.agents import DQNAgent

env = gym.make('KungFuMaster-ram-v0')
actions = env.action_space.n
model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape ))

model.add(Dense(16))

model.add(Activation('relu'))

model.add(Dense(actions))

model.add(Activation('linear'))

print(model.summary())
policy = EpsGreedyQPolicy()



memory = SequentialMemory(50000, window_length = 1)



dqn = DQNAgent(model = model, nb_actions = actions, memory = memory, nb_steps_warmup = 10, policy=policy)
dqn.compile(Adam(lr=1e-3),metrics=['mae'])
dqn.fit(env,nb_steps=1500, visualize=False, verbose=2)
dqn.test(env,nb_episodes=30,visualize=False)