import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import time
import cv2
import tensorflow as tf
import random as rand
from collections import deque
!pip install gym[atari] 
GAME = "CartPole-v0"
env = gym.envs.make(GAME)
print("Action space: {}".format(env.action_space))
print("Action space size: {}".format(env.action_space.n))
observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

print("-"*10)
action = env.action_space.sample()
print('Take action {}'.format(action))
observation, reward, game_over, info = env.step(action)
print("observation: {}, reward: {}, game_over: {}, info: {} ".format(observation.shape, reward, game_over, info))

env.close()
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # deque is actually implemented as linked list, so this is a suboptimal solution for random sampling. A custom ring buffer would be better.
        # However, for education purposes this will suffice.
        self.replay_memory = deque(maxlen=buffer_size)    

    def add(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size <= len(self.replay_memory):
            return rand.sample(self.replay_memory, batch_size)
        else:
            assert False

    def __len__(self):
        return len(self.replay_memory)
class LinearSchedule():
    def __init__(self, start_epsilon=1, final_epsilon=0.1, pre_train_steps=10, final_exploration_step=100):
        self.pre_train_steps = pre_train_steps
        self.final_exploration_step = final_exploration_step
        self.final_epsilon = final_epsilon
        self.decay_factor = self.pre_train_steps/self.final_exploration_step
        self.epsilon = self.pre_train_steps * (1-self.decay_factor) + self.final_exploration_step * self.decay_factor
    
    def value(self, t):
        if t > self.pre_train_steps:
            self.decay_factor = (t - self.pre_train_steps)/self.final_exploration_step
            self.epsilon = 1-self.decay_factor
            self.epsilon = max(self.final_epsilon, self.epsilon)
            return self.epsilon
        else:
            return 1
class DQN(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.hidden_layers = []
        self.hidden_layers.append(tf.keras.layers.Dense(64, activation='relu'))
        self.hidden_layers.append(tf.keras.layers.Dense(32, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(units=num_actions, activation='linear')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for l in self.hidden_layers:
            z = l(z)
        q_vals = self.output_layer(z)
        return q_vals
class Agent:
    def __init__(self, epsilon_schedule, replay_buffer, num_actions=2, gamma=0.9, batch_size=64, lr=0.001,
                 max_episodes = 500, max_steps_per_episode=2000, steps_until_sync=20, choose_action_frequency=1,
                 pre_train_steps = 1, train_frequency=1):
        
        # dqn is used to predict Q-values to decide which action to take
        self.dqn = DQN([4], num_actions)
        self.dqn.build(tf.TensorShape([None, 4]))
        
        # dqn_target is used to predict the future reward
        self.dqn_target = DQN([4], num_actions)
        self.dqn_target.build(tf.TensorShape([None, 4]))

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.epsilon_schedule = epsilon_schedule
        self.steps_until_sync = steps_until_sync
        self.choose_action_frequency = choose_action_frequency
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.train_frequency = train_frequency
        self.loss_function = tf.keras.losses.MSE
        self.pre_train_steps = pre_train_steps
        
        self.episode_reward_history = []

    def predict_q(self, inputs):
        return self.dqn(inputs)

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            # explore
            return np.random.choice(self.num_actions)
        else:
            # exploit
            return np.argmax(self.predict_q(np.expand_dims(states, axis=0))[0])

    def update_target_network(self):
        self.dqn_target.set_weights(self.dqn.get_weights())

    def train_step(self):
        mini_batch = self.replay_buffer.sample(self.batch_size)

        observations_batch, action_batch, reward_batch, next_observations_batch, done_batch = map(np.array,
                                                                                                  zip(*mini_batch))

        with tf.GradientTape() as tape:
            dqn_variables = self.dqn.trainable_variables
            tape.watch(dqn_variables)

            future_rewards = self.dqn_target(tf.convert_to_tensor(next_observations_batch, dtype=tf.float32))
            next_action = tf.argmax(future_rewards, axis=1)
            target_q = tf.reduce_sum(tf.one_hot(next_action, self.num_actions) * future_rewards, axis=1)
            target_q = (1 - done_batch) * self.gamma * target_q + reward_batch

            predicted_q = self.dqn(tf.convert_to_tensor(observations_batch, dtype=tf.float32))
            predicted_q = tf.reduce_sum(tf.one_hot(action_batch, self.num_actions) * predicted_q, axis=1)
            loss = self.loss_function(target_q, predicted_q)
            
        # Backprop
        gradients = tape.gradient(loss, dqn_variables)
        self.optimizer.apply_gradients(zip(gradients, dqn_variables))
        
        return loss

    def train(self, env):
        episode = 0
        total_step = 0
        episode_step = 0
        state = env.reset()
        loss = 0
        last_hundred_rewards = deque(maxlen=100)

        while episode < self.max_episodes:
            current_state = env.reset()
            done = False
            action = 0
            episode_reward = 0
            episode_step = 0
            epsilon = epsilon_schedule.value(total_step)

            while not done:
                if total_step % self.choose_action_frequency == 0:
                    if len(replay_buffer) > self.batch_size:
                        action = self.get_action(current_state, epsilon)
                    else:
                        action = self.get_action(current_state, 1.0)

                next_state, reward, done, info = env.step(action)
                
                self.replay_buffer.add(current_state, action, reward, next_state, done)
                episode_reward += reward

                if total_step > self.pre_train_steps and len(replay_buffer) > self.batch_size:
                    loss = self.train_step()

                if total_step % self.steps_until_sync == 0:
                    self.update_target_network()
                                    
                #end of step
                total_step += 1
                episode_step += 1
                current_state = next_state
                
            # end of episode
            self.episode_reward_history.append(episode_reward)
            last_hundred_rewards.append(episode_reward)
            mean_episode_reward = np.mean(last_hundred_rewards)
            
            if episode % 50 == 0:
                print(f'Episode {episode} (Step {total_step}) - Moving Avg Reward: {mean_episode_reward:.3f} Loss: {loss:.5f} Epsilon: {epsilon:.3f}')

            if mean_episode_reward >= 195:
                print(f'Task solved after {episode} episodes! (Moving Avg Reward: {mean_episode_reward:.3f})')
                return
                
            episode += 1
            

env = gym.envs.make(GAME)

epsilon_schedule = LinearSchedule(start_epsilon=1, final_epsilon=0.1, pre_train_steps=100, final_exploration_step=10_000)

replay_buffer = ReplayBuffer(32_000)

agent = Agent(epsilon_schedule, replay_buffer, num_actions=2, gamma=0.99, batch_size=64, lr=0.0007,
                 max_episodes=3000, steps_until_sync=200, choose_action_frequency=1)
agent.train(env)

env.close()
