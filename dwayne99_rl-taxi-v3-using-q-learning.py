import numpy as np

import gym

import random



from tqdm import tqdm
env = gym.make('Taxi-v3')

env.render()
action_size = env.action_space.n

print(f"Total possible actions: {action_size}")



state_size = env.observation_space.n

print(f"Total states: {state_size}")
qtable = np.zeros((state_size,action_size))

print(qtable)
total_episodes = 50000       # Total episodes to train the agent for

total_test_episodes = 10     # Number of test episodes

max_steps = 99               # Terminate if the agent takes more than 99 steps



alpha = 0.7                  # Learning rate

gamma = 0.618                # Discounting rate for rewards



# parameters for maintaining trade-off between exploration-exploitation

epsilon = 1.0                # Exploration rate

max_epsilon = 1.0            # Exploration probability at the start

min_epsilon = 0.01           # Minimum exploration probability

decay_rate = 0.01            # rate at which epsilon shrinks 
# iterate over every episode

for episode in tqdm(range(total_episodes)):

    

    # Reset the environment at every episode

    state = env.reset()

    # flag to check if episode is terminated or not

    done = False

    

    # iterate over all steps that the agent can take in an episode

    for step in range(max_steps): 

        

        # Select an action based on the epsilon-greedy policy

        

        # probability to select exploitation

        one_minus_epsilon = random.uniform(0,1)

        

        # if one_minus_epsilon is greater than epsilon then exploit 

        if one_minus_epsilon > epsilon:

            action = np.argmax(qtable[state,:])

        # else explore by selecting an action randomly

        else:

            action = env.action_space.sample()

            

        # Take this action to reach the next state and get a reward 

        new_state, reward, done, info = env.step(action)

        

        # update the Q-Table based on the formula given in the algorithm

        qtable[state,action] = qtable[state,action] + alpha*(reward + gamma*np.max(qtable[new_state,:]) - qtable[state,action])

        

        # update the current state

        state = new_state

        

        # if the agent has reached termination state then break

        if done:

            break

        

    # epsilon decay to maintain trade-off between exploration-exploitation

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
# keep track of all rewards

rewards = []



for episode in range(total_test_episodes):

    

    state = env.reset()

    done = False

    total_rewards = 0

    print(f"{'*'*80}")

    print(f"Episode {episode + 1}:")

    

    for step in range(max_steps):

        

        # render every frame of the agent

        env.render()

        

        # take an action that has max expected future reward given in that state

        action = np.argmax(qtable[state,:])

        

        new_state, reward, done, info = env.step(action)

        

        total_rewards += reward

        

        if done:

            # keep track of rewards received at every episode

            rewards.append(total_rewards)

            print(f"Score: {total_rewards}")

            break

        

        state = new_state



env.close()

print(f"Average Rewards: {sum(rewards)/total_test_episodes}")