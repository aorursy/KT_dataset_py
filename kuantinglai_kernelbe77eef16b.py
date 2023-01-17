import numpy as np
import random
import gym

env = gym.make("FrozenLake8x8-v0") 
env.reset()
env.render()
actions = env.action_space.n
states = env.observation_space.n
eposides = 100000
epsilon = 0.8
gamma = 0.9
alpha = 0.01
# Create Q table with all rewards = 0
q_table = np.zeros((states, actions))
# Training
for i in range(eposides):
    env.reset()
    done = False
    state = 0
    steps = 0
    total_reward = 0
    while not done:
        # epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[state,:]) # Exploit
        
        # Move one step
        next_state, reward, done, _ = env.step(action)
        
        # Update Q table
        q_table[state, action] = q_table[state, action] + alpha*(reward + gamma*np.max(q_table[next_state, :]) - q_table[state, action])
        state = next_state
        
        # Update statistics
        steps = steps + 1
        total_reward = total_reward + reward
    
    print("Episode {}, steps {}, reward {}".format(i, steps, total_reward))
q_table # Q table after learning
# Testing: Calculating the average reward of 1000 eposides
test_episodes = 1000 # DON'T CHANGE THIS VALUE
steps = 0
total_reward = 0
for i in range(test_episodes):
    env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state,:])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        steps = steps + 1
        total_reward = total_reward + reward
    
print("The average results of {} episodes are steps {}, reward {}".format(test_episodes, steps/test_episodes, total_reward/test_episodes))
total_avg_reward = total_reward/test_episodes
# Print results in CSV format and upload to Kaggle
with open('rewards.csv', 'w') as f:
    f.write('Id,Predicted\n')
    f.write('FrozenLake8x8_public,{}\n'.format(total_avg_reward))
    f.write('FrozenLake8x8_private,{}\n'.format(total_avg_reward))
# Download your results!
from IPython.display import FileLink
FileLink('rewards.csv')