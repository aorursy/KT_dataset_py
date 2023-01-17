import gym
import torch

import time

import matplotlib.pyplot as plt


from gym.envs.registration import register

register(

    id='FrozenLakeNotSlippery-v0',

    entry_point='gym.envs.toy_text:FrozenLakeEnv',

    kwargs={'map_name' : '4x4', 'is_slippery': False},

)



env = gym.make('FrozenLakeNotSlippery-v0')



# Instantiate the Environment.

# env = gym.make('FrozenLake-v0')



# To check all environments present in OpenAI

# print(envs.registry.all())





env.render()
# Total number of States and Actions

number_of_states = env.observation_space.n

number_of_actions = env.action_space.n

print( "States = ", number_of_states)

print( "Actions = ", number_of_actions)



num_episodes = 1000

steps_total = []

rewards_total = []

egreedy_total = []

# if learning_rate == 0:

#      Pick value of new Q(s,a) based on past experience

# elif learning_rate == 1:

#      Pick value of new Q(s,a) based on current situtation



# Value of learning_rate(alpha) varies from [0 - 1]
# Discount rate accounts for the Reward the agent receive on an action



# if discount_rate == 0:

#     only current reward accounted

# elif discount_rate == 1:

#     future rewards also accounted

    
# PARAMS 



# Discount on reward

gamma = 0.95



# Factor to balance the ratio of action taken based on past experience to current situtation

learning_rate = 0.9



# exploit vs explore to find action

# Start with 70% random actions to explore the environment

# And with time, using decay to shift to more optimal actions learned from experience



egreedy = 0.7

egreedy_final = 0.1

egreedy_decay = 0.999
Q = torch.zeros([number_of_states, number_of_actions])

Q

for i_episode in range(num_episodes):

    

    # resets the environment

    state = env.reset()

    step = 0



    while True:

        

        step += 1

        

        random_for_egreedy = torch.rand(1)[0]

        



        if random_for_egreedy > egreedy:      

            random_values = Q[state] + torch.rand(1,number_of_actions) / 1000      

            action = torch.max(random_values,1)[1][0]  

            action = action.item()

        else:

            action = env.action_space.sample()

            

        if egreedy > egreedy_final:

            egreedy *= egreedy_decay

        

        new_state, reward, done, info = env.step(action)



        # Filling the Q Table

        Q[state, action] = reward + gamma * torch.max(Q[new_state])

        

        # Setting new state for next action

        state = new_state

        

        # env.render()

        # time.sleep(0.4)

        

        if done:

            steps_total.append(step)

            rewards_total.append(reward)

            egreedy_total.append(egreedy)

            if i_episode % 10 == 0:

                print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode,reward, step))

            break

        



print(Q)

        

print("Percent of episodes finished successfully: {0}".format(sum(rewards_total)/num_episodes))

print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(rewards_total[-100:])/100))



print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))

print("Average number of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))

plt.title("Rewards")

plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)

plt.show()



plt.figure(figsize=(12,5))

plt.title("Steps / Episode length")

plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)

plt.show()



plt.figure(figsize=(12,5))

plt.title("Egreedy value")

plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue', width=5)

plt.show()
