import matplotlib.pyplot as plt

import numpy as np

import random

import matplotlib.pyplot as plt



from IPython import display

from IPython.display import clear_output

from time import sleep



%matplotlib inline
!pip install gym
import gym
env = gym.make("Taxi-v3").env
env.render()
print(env.action_space.n)
print(env.observation_space.n)
action = env.action_space.sample()

print(action)
env.reset() # remet l'environnement dans l'état initial

env.render()

action = env.action_space.sample()

state,reward,done,info = env.step(action)

env.render()
print(state,reward,done,info)
env.render()
env.s
def policy() :

    action = env.action_space.sample()

    return action
def play_episode(state) :

    done = False

    env.s = state

    while not done :

        env.render()

        action = policy()

        _,_,done,_ = env.step(action)

    env.close()
state = env.reset()
episode = play_episode(state)
Q = np.random.rand(env.observation_space.n, env.action_space.n)
print(Q)
def policy(state) :

    epsilon = 0.5

    if random.uniform(0, 1) < epsilon :

        action = env.action_space.sample() # Choix aléatoire de la prochaine action

    else:

        action = np.argmax(Q[state]) # Choix de la meilleure action dans la table Q

    return action
alpha = 0.6

gamma = 0.9
def play_episode(state) :

    done = False

    env.s = state

    total_reward, nsteps = 0, 0

    while not done :

        action = policy(state)

        next_state,reward,done,_ = env.step(action)

# Compléter ici avec la mise à jour de la table Q

# Indication : np.max(Q[s]) donne le max de la ligne correspondant à l'état s dans la table Q

        state = next_state

        total_reward += reward

        nsteps += 1

        if nsteps > 200 : break

    return total_reward, nsteps

    
nb_episodes = 1000



for i in range(nb_episodes) :

    state = env.reset()

    total_reward, nsteps = play_episode(state)

    print("Episode n° ",i)

    clear_output(wait=True)



print(total_reward, nsteps)
def epsilon(n) :

    return 1/np.sqrt(1+float(n))
plt.plot([epsilon(n) for n in range(1000)])
def policy(state, n) :

# Compléter la fonction

    return action
def play_episode(state, episode_i) :

    done = False

    env.s = state

    total_reward, nsteps = 0, 0

    while not done :

        action = policy(state, episode_i)

        next_state,reward,done,_ = env.step(action)

# Compléter

        state = next_state

        total_reward += reward

        nsteps += 1

        if nsteps > 200 : break

    return total_reward, nsteps

    
def fit(nb_episodes) :

    list_rewards = [] # Liste des résultats

    list_steps = [] # Liste du nombre d'étapes par épisode

    for i in range(nb_episodes) :

        state = env.reset()

        total_reward, nsteps = play_episode(state,i)

        print("Episode n° ",i)

        clear_output(wait=True)

        list_rewards.append(total_reward)

        list_steps.append(nsteps) 

    

    return list_rewards, list_steps

    
Q = np.random.rand(env.observation_space.n, env.action_space.n)



alpha = 0.3

gamma = 0.6



list_rewards, list_steps = fit(1000)
plt.plot(list_rewards, 'orange')

plt.ylabel('Total reward')

plt.xlabel('Episode')
plt.plot(list_steps, 'cyan')

plt.ylabel('Nombre d étapes')

plt.xlabel('Episode')