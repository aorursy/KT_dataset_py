!apt-get install -y xvfb python-opengl x11-utils

!apt-get install -y --no-install-recommends ffmpeg

!pip install ffmpeg

!pip install gym pyvirtualdisplay scikit-video #> /dev/null 2>&1

#!pip install 'gym[box2d]'

!pip install atari_py



#%tensorflow_version 2.x

import tensorflow as tf



import numpy as np

import base64, io, time, gym

import IPython, functools

import matplotlib.pyplot as plt

from tqdm import tqdm



!pip install mitdeeplearning

import mitdeeplearning as mdl
### Instantiate the Cartpole environment ###



env = gym.make("CartPole-v0")

env.seed(1)
n_observations = env.observation_space

print("Environment has observation space =", n_observations)
n_actions = env.action_space.n

print("Number of possible actions that the agent can choose from =", n_actions)
### Define the Cartpole agent ###



# Defines a feed-forward neural network

def create_cartpole_model():

  model = tf.keras.models.Sequential([

      # First Dense layer

      tf.keras.layers.Dense(units=32, activation='relu'),



      # TODO: Define the last Dense layer, which will provide the network's output.

      # Think about the space the agent needs to act in!

      tf.keras.layers.Dense(units=n_actions, activation=None)

  ])

  return model



cartpole_model = create_cartpole_model()
### Define the agent's action function ###



# Function that takes observations as input, executes a forward pass through model, 

#   and outputs a sampled action.

# Arguments:

#   model: the network that defines our agent

#   observation: observation which is fed as input to the model

# Returns:

#   action: choice of agent action

def choose_action(model, observation):

  # add batch dimension to the observation

  observation = np.expand_dims(observation, axis=0)



  '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''

  logits = model.predict(observation)

  

  # pass the log probabilities through a softmax to compute true probabilities

  prob_weights = tf.nn.softmax(logits).numpy()

  '''TODO: randomly sample from the prob_weights to pick an action.

  Hint: carefully consider the dimensionality of the input probabilities (vector) and the output action (scalar)'''

  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]



  return action
observation = env.reset()

action = choose_action(cartpole_model, observation)

print("Random action from first observation: ", action)
### Agent Memory ###



class Memory:

  def __init__(self): 

      self.clear()



  # Resets/restarts the memory buffer

  def clear(self): 

      self.observations = []

      self.actions = []

      self.rewards = []



  # Add observations, actions, rewards to memory

  def add_to_memory(self, new_observation, new_action, new_reward): 

      self.observations.append(new_observation)

      self.actions.append(new_action)

      self.rewards.append(new_reward)

  

  def print(self):

      print(f'The memory observations {self.observations}')

      print(f'The memory actions {self.actions}')

      print(f'The memory rewards {self.rewards}')

        

memory = Memory()
observation = env.reset()

action = choose_action(cartpole_model, observation)

next_observation, reward, done, info = env.step(action)

memory.add_to_memory(next_observation, action, reward)

memory.print()
### Reward function ###



# Helper function that normalizes an np.array x

def normalize(x):

  x -= np.mean(x)

  x /= np.std(x)

  return x.astype(np.float32)



# Compute normalized, discounted, cumulative rewards (i.e., return)

# Arguments:

#   rewards: reward at timesteps in episode

#   gamma: discounting factor

# Returns:

#   normalized discounted reward

def discount_rewards(rewards, gamma=0.95): 

  discounted_rewards = np.zeros_like(rewards)

  R = 0

  for t in reversed(range(0, len(rewards))):

      # update the total discounted reward

      R = R * gamma + rewards[t]

      discounted_rewards[t] = R



  return normalize(discounted_rewards)
observation = env.reset()

memory.clear()

for i in range(10):

  action = choose_action(cartpole_model, observation)

  next_observation, reward, done, info = env.step(action)

  memory.add_to_memory(next_observation, action, reward)



memory.print()



discounted_rewards = discount_rewards(memory.rewards)

print("Rewards: ", memory.rewards)

print("Discounted rewards: ", discounted_rewards)
### Loss function ###



# Arguments:

#   logits: network's predictions for actions to take

#   actions: the actions the agent took in an episode

#   rewards: the rewards the agent received in an episode

# Returns:

#   loss

def compute_loss(logits, actions, rewards): 

  '''TODO: complete the function call to compute the negative log probabilities'''

  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)

  

  '''TODO: scale the negative log probability by the rewards'''

  loss = tf.reduce_mean(neg_logprob * rewards)

  return loss
observation = env.reset()

memory.clear()

for i in range(10):

  action = choose_action(cartpole_model, observation)

  next_observation, reward, done, info = env.step(action)

  memory.add_to_memory(next_observation, action, reward)



memory.print()

logits = cartpole_model.predict(np.asarray(memory.observations))

print("Logits: ", logits)

loss = compute_loss(logits, memory.actions, memory.rewards)

print("Tensor loss:", loss)
### Training step (forward and backpropagation) ###



def train_step(model, optimizer, observations, actions, discounted_rewards):

  with tf.GradientTape() as tape:

      # Forward propagate through the agent network

      logits = model(observations)



      '''TODO: call the compute_loss function to compute the loss'''

      loss = compute_loss(logits, actions, discounted_rewards)



  '''TODO: run backpropagation to minimize the loss using the tape.gradient method.

      Use `model.trainable_variables`'''

  grads = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(grads, model.trainable_variables))

learning_rate = 1e-3

optimizer = tf.keras.optimizers.Adam(learning_rate)



observation = env.reset()

memory.clear()

for i in range(5):

  action = choose_action(cartpole_model, observation)

  next_observation, reward, done, info = env.step(action)

  memory.add_to_memory(next_observation, action, reward)



memory.print()

train_step(cartpole_model, optimizer, np.asarray(memory.observations), np.asarray(memory.actions), discount_rewards(memory.rewards))

### Cartpole training! ###



# Learning rate and optimizer

learning_rate = 1e-3

optimizer = tf.keras.optimizers.Adam(learning_rate)



# instantiate cartpole agent

cartpole_model = create_cartpole_model()



# to track our progress

smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)

plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')



if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for i_episode in range(500):



  plotter.plot(smoothed_reward.get())



  # Restart the environment

  observation = env.reset()

  memory.clear()



  while True:

      # using our observation, choose an action and take it in the environment

      action = choose_action(cartpole_model, observation)

      next_observation, reward, done, info = env.step(action)

      # add to memory

      memory.add_to_memory(observation, action, reward)

      

      # is the episode over? did you crash or do so well that you're done?

      if done:

          # determine total reward and keep a record of this

          total_reward = sum(memory.rewards)

          smoothed_reward.append(total_reward)

          

          # initiate training - remember we don't know anything about how the 

          #   agent is doing until it has crashed!

          train_step(cartpole_model, optimizer, 

                     observations=np.vstack(memory.observations),

                     actions=np.array(memory.actions),

                     discounted_rewards = discount_rewards(memory.rewards))

          

          # reset the memory

          memory.clear()

          break

      # update our observatons

      observation = next_observation
saved_cartpole = mdl.lab3.save_video_of_model(cartpole_model, "CartPole-v0")

mdl.lab3.play_video(saved_cartpole)
env = gym.make("Pong-v0", frameskip=5)

env.seed(1); # for reproducibility
print("Environment has observation space =", env.observation_space)
n_actions = env.action_space.n

print("Number of possible actions that the agent can choose from =", n_actions)
### Define the Pong agent ###



# Functionally define layers for convenience

# All convolutional layers will have ReLu activation

Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')

Flatten = tf.keras.layers.Flatten

Dense = tf.keras.layers.Dense



# Defines a CNN for the Pong agent

def create_pong_model():

  model = tf.keras.models.Sequential([

    # Convolutional layers

    # First, 16 7x7 filters with 4x4 stride

    Conv2D(filters=16, kernel_size=7, strides=4),



    # TODO: define convolutional layers with 32 5x5 filters and 2x2 stride

    Conv2D(filters=32, kernel_size=5, strides=2),



    # TODO: define convolutional layers with 48 3x3 filters and 2x2 stride

    Conv2D(filters=48, kernel_size=3, strides=2),



    Flatten(),

    

    # Fully connected layer and output

    Dense(units=64, activation='relu'),

    # TODO: define the output dimension of the last Dense layer. 

    # Pay attention to the space the agent needs to act in

    Dense(units=n_actions, activation=None)

  

  ])

  return model



pong_model = create_pong_model()
### Pong reward function ###



# Compute normalized, discounted rewards for Pong (i.e., return)

# Arguments:

#   rewards: reward at timesteps in episode

#   gamma: discounting factor. Note increase to 0.99 -- rate of depreciation will be slower.

# Returns:

#   normalized discounted reward

def discount_rewards(rewards, gamma=0.99): 

  discounted_rewards = np.zeros_like(rewards)

  R = 0

  for t in reversed(range(0, len(rewards))):

      # NEW: Reset the sum if the reward is not 0 (the game has ended!)

      if rewards[t] != 0:

        R = 0

      # update the total discounted reward as before

      R = R * gamma + rewards[t]

      discounted_rewards[t] = R

      

  return normalize(discounted_rewards)
observation = env.reset()

for i in range(30):

  observation, _,_,_ = env.step(0)

observation_pp = mdl.lab3.preprocess_pong(observation)



f = plt.figure(figsize=(10,3))

ax = f.add_subplot(121)

ax2 = f.add_subplot(122)

ax.imshow(observation); ax.grid(False);

ax2.imshow(np.squeeze(observation_pp)); ax2.grid(False); plt.title('Preprocessed Observation');
### Training Pong ###



# Hyperparameters

learning_rate=1e-4

MAX_ITERS = 500 # increase the maximum number of episodes, since Pong is more complex!



# Model and optimizer

pong_model = create_pong_model()

optimizer = tf.keras.optimizers.Adam(learning_rate)



# plotting

smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)

plotter = mdl.util.PeriodicPlotter(sec=5, xlabel='Iterations', ylabel='Rewards')

memory = Memory()



for i_episode in range(MAX_ITERS):



  plotter.plot(smoothed_reward.get())



  # Restart the environment

  observation = env.reset()

  previous_frame = mdl.lab3.preprocess_pong(observation)



  while True:

      # Pre-process image 

      current_frame = mdl.lab3.preprocess_pong(observation)

      

      '''TODO: determine the observation change

      Hint: this is the difference between the past two frames'''

      obs_change = current_frame - previous_frame

      

      '''TODO: choose an action for the pong model, using the frame difference, and evaluate'''

      action = choose_action(pong_model, obs_change)

      # Take the chosen action

      next_observation, reward, done, info = env.step(action)



      '''TODO: save the observed frame difference, the action that was taken, and the resulting reward!'''

      memory.add_to_memory(obs_change, action, reward)

      

      # is the episode over? did you crash or do so well that you're done?

      if done:

          # determine total reward and keep a record of this

          total_reward = sum(memory.rewards)

          smoothed_reward.append( total_reward )



          # begin training

          train_step(pong_model, 

                     optimizer, 

                     observations = np.stack(memory.observations, 0), 

                     actions = np.array(memory.actions),

                     discounted_rewards = discount_rewards(memory.rewards))

          

          memory.clear()

          break



      observation = next_observation

      previous_frame = current_frame
saved_pong = mdl.lab3.save_video_of_model(

    pong_model, "Pong-v0", obs_diff=True, 

    pp_fn=mdl.lab3.preprocess_pong)

mdl.lab3.play_video(saved_pong)