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

state,reward,done,info = env.step(action) # l'état après l'exécution

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
Q = np.random.rand(env.observation_space.n, env.action_space.n) # Initialiser le tableau
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

        

        value_ancien = Q[state,action]

        max_proch = np.max(Q[next_state])

        value_nouv = (1-alpha) * value_ancien + alpha * (reward + gamma * max_proch)

        Q[state,action] = value_nouv

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

    if random.uniform(0, 1) < epsilon(n) :

        action = env.action_space.sample() # Choix aléatoire de la prochaine action

    else:

        action = np.argmax(Q[state]) # Choix de la meilleure action dans la table Q

    return action
def play_episode(state, episode_i) :

    done = False

    env.s = state

    total_reward, nsteps = 0, 0

    while not done :

        action = policy(state, episode_i)

        next_state,reward,done,_ = env.step(action)

        

        value_ancien = Q[state,action]

        max_proch = np.max(Q[next_state])

        value_nouv = (1-alpha) * value_ancien + alpha * (reward + gamma * max_proch)

        Q[state,action] = value_nouv

        

        state = next_state

        total_reward += reward

        nsteps += 1

        if nsteps > 200 : break

    return total_reward, nsteps, action

    
def fit(nb_episodes) :

    list_rewards = [] # Liste des résultats

    list_steps = [] # Liste du nombre d'étapes par épisode

    frames=[];

    

    for i in range(nb_episodes) :

        state = env.reset()

        total_reward, nsteps, action = play_episode(state,i)

        print("Episode n° ",i)

        clear_output(wait=True)

        list_rewards.append(total_reward)

        list_steps.append(nsteps) 

        

        frames.append({

            'frame': env.render(mode='ansi'),

            'state': state,

            'action': action,

            'reward': total_reward

        }

        )

    

    return list_rewards, list_steps,frames

    
Q = np.random.rand(env.observation_space.n, env.action_space.n)



alpha = 0.3

gamma = 0.6



list_rewards, list_steps,frames = fit(1000)
from IPython.display import clear_output

from time import sleep

from io import StringIO



def print_frames(frames):

    for i, frame in enumerate(frames):

        clear_output(wait=True)

        print(frame['frame'])

        print(f"Timestep: {i + 1}")

        print(f"State: {frame['state']}")

        print(f"Action: {frame['action']}")

        print(f"Reward: {frame['reward']}")

        sleep(.1)

        

print_frames(frames)
plt.plot(list_rewards, 'orange')

plt.ylabel('Total reward')

plt.xlabel('Episode')
plt.plot(list_steps, 'cyan')

plt.ylabel('Nombre d étapes')

plt.xlabel('Episode')