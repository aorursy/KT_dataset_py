import sys

import gym

import numpy as np

import random

import math

from collections import defaultdict, deque

import matplotlib.pyplot as plt

%matplotlib inline



# utility scrips with some helper functions

from rl_plot_utils import plot_values
# Create the environment

env = gym.make('CliffWalking-v0')
# Number of Actions that the agent can take

print(env.action_space)



# Number of states the agent could be in 

print(env.observation_space)
# define the optimal state-value function

V_opt = np.zeros((4,12))

V_opt[0][0:13] = -np.arange(3, 15)[::-1]

V_opt[1][0:13] = -np.arange(3, 15)[::-1] + 1

V_opt[2][0:13] = -np.arange(3, 15)[::-1] + 2

V_opt[3][0] = -13



plot_values(V_opt)
def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state = None, next_action = None):

    '''Returns updated Q-value for the most recent experience'''

    

    # get the current state-action from the Q-table

    current = Q[state][action]

    

    # get the next state-action from the Q-table

    Qsa_next = Q[next_state][next_action] if next_state is not None else 0

    

    new_value = current + alpha*(reward + gamma*(Qsa_next) - current)

    

    return new_value
def epsilon_greedy(Q, state, nA, eps):

    '''

    Selects epsilon-greedy action for the supplied state

    

    parameters:

    ===========

        Q (dictionary): action-value function

        state (int)   : current state

        nA (int)      : number of actions in the env

        eps (float)   : epsilon

    '''

    

    # exploitation-exploration step

    if random.random() > eps:

        # select the greedy action

        return np.argmax(Q[state])

    else:

        # select a random action

        return random.choice(np.arange(env.action_space.n))
def sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):

    

    nA = env.action_space.n # num of actions

    

    # initialize empty dict of arrays

    Q = defaultdict(lambda: np.zeros(nA))

    

    # monitor performance

    tmp_scores = deque(maxlen=plot_every) # deque for keeping track of scores

    

    avg_scores = deque(maxlen=num_episodes) # average scores over every plot_every episodes

    

    for e in range(1, num_episodes + 1):

        

        if e % 1000 == 0:

            print(f"Episode {e}")

            sys.stdout.flush()

        

        # keep track of the score for each episiode

        score = 0 

        

        # start episode

        state = env.reset()

        

        # set value for epsilon (decaying)

        eps = 1.0 / e

        

        # perform the first action

        action = epsilon_greedy(Q, state, nA, eps)

        

        while True:

            

            # take action 'a' and obverse state 's' and reward 'r'

            next_state, reward, done, info = env.step(action)

            

            # sum up the the reward at every step in an episode

            score += reward

            

            # check if the episode reached termination

            if not done:

                

                # get the next epsilon-greedy action

                next_action = epsilon_greedy(Q, next_state, nA, eps)

                

                # update the Q-Table

                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state, next_action)

                

                # update the new current state and action

                state = next_state

                action = next_action

            

            else:

                # update the Q-table

                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state, next_action)

                

                # append the score

                tmp_scores.append(score)

                break

                

        if (e % plot_every == 0):

            avg_scores.append(np.mean(tmp_scores))

                

    # plot performance

    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))

    plt.xlabel('Episode Number')

    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)

    plt.show()

    # print best 100-episode performance

    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))    

    return Q                  
# obtain the estimated optimal policy and corresponding action-value function

Q_sarsa = sarsa(env, 5000, .01)



# print the estimated optimal policy

policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)

print(policy_sarsa)



# plot the estimated optimal state-value function

V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])

plot_values(V_sarsa)
def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):

    """Returns updated Q-value for the most recent experience."""

    

    current = Q[state][action]  # estimate in Q-table (for current state, action pair)

    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state 

    

    # get the new value in accordance to the formula mentioned above

    new_value = current + (alpha * (reward + (gamma * Qsa_next) - current)) # get updated value 

    return new_value
def Q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):

    """Q-Learning - TD Control

    

    Params

    ======

        num_episodes (int): number of episodes to run the algorithm

        alpha (float)     : learning rate

        gamma (float)     : discount factor

        plot_every (int)  : number of episodes to use when calculating average score

    """

    

    nA = env.action_space.n # num of actions

    

    # initialize empty dict of arrays

    Q = defaultdict(lambda: np.zeros(nA))

    

    # monitor performance

    tmp_scores = deque(maxlen=plot_every) # deque for keeping track of scores

    

    avg_scores = deque(maxlen=num_episodes) # average scores over every plot_every episodes

    

    for e in range(1, num_episodes + 1):

        

        if e % 1000 == 0:

            print(f"Episode {e}")

            sys.stdout.flush()

        

        # keep track of the score for each episiode

        score = 0 

        

        # start episode

        state = env.reset()

        

        # set value for epsilon (decaying)

        eps = 1.0 / e

        

        # perform the first action

        action = epsilon_greedy(Q, state, nA, eps)

        

        while True:

            

            # take action 'a' and obverse state 's' and reward 'r'

            next_state, reward, done, info = env.step(action)

            

            # sum up the the reward at every step in an episode

            score += reward

            

            # check if the episode reached termination

            if not done:

                

                # get the next epsilon-greedy action

                next_action = epsilon_greedy(Q, next_state, nA, eps)

                

                # update the Q-Table

                Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state)

                

                # update the new current state and action

                state = next_state

                action = next_action

            

            else:

                # update the Q-table

                Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state)

                

                # append the score

                tmp_scores.append(score)

                break

                

        if (e % plot_every == 0):

            avg_scores.append(np.mean(tmp_scores))

                

    # plot performance

    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))

    plt.xlabel('Episode Number')

    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)

    plt.show()

    # print best 100-episode performance

    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))    

    return Q   
# obtain the estimated optimal policy and corresponding action-value function

Q_sarsamax = Q_learning(env, 5000, .01)



# print the estimated optimal policy

policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))



print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")

print(policy_sarsamax)



# plot the estimated optimal state-value function

plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):

    """Returns updated Q-value for the most recent experience."""

    

    current = Q[state][action]         # estimate in Q-table (for current state, action pair)

    policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')

    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA) # greedy action

    Qsa_next = np.dot(Q[next_state], policy_s)         # get value of state at next time step

    target = reward + (gamma * Qsa_next)               # construct target

    new_value = current + (alpha * (target - current)) # get updated value 

    

    return new_value


def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):

    """Expected SARSA - TD Control

    

    Params

    ======

        num_episodes (int): number of episodes to run the algorithm

        alpha (float): step-size parameters for the update step

        gamma (float): discount factor

        plot_every (int): number of episodes to use when calculating average score

    """

     

    nA = env.action_space.n # num of actions

    

    # initialize empty dict of arrays

    Q = defaultdict(lambda: np.zeros(nA))

    

    # monitor performance

    tmp_scores = deque(maxlen=plot_every) # deque for keeping track of scores

    

    avg_scores = deque(maxlen=num_episodes) # average scores over every plot_every episodes

    

    for e in range(1, num_episodes + 1):

        

        if e % 1000 == 0:

            print(f"Episode {e}")

            sys.stdout.flush()

        

        # keep track of the score for each episiode

        score = 0 

        

        # start episode

        state = env.reset()

        

        # set value for epsilon (decaying)

        eps = 0.005

        

        # perform the first action

        action = epsilon_greedy(Q, state, nA, eps)

        

        while True:

            

            # take action 'a' and obverse state 's' and reward 'r'

            next_state, reward, done, info = env.step(action)

            

            # sum up the the reward at every step in an episode

            score += reward

            

            # check if the episode reached termination

            if not done:

                

                # get the next epsilon-greedy action

                next_action = epsilon_greedy(Q, next_state, nA, eps)

                

                # update the Q-Table

                Q[state][action] = update_Q_expsarsa(alpha, gamma,nA,eps, Q, state, action, reward, next_state)

                

                # update the new current state and action

                state = next_state

                action = next_action

            

            else:

                # update the Q-table

                Q[state][action] = update_Q_expsarsa(alpha, gamma,nA,eps, Q, state, action, reward, next_state)

                

                # append the score

                tmp_scores.append(score)

                break

                

        if (e % plot_every == 0):

            avg_scores.append(np.mean(tmp_scores))

                

    # plot performance

    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))

    plt.xlabel('Episode Number')

    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)

    plt.show()

    # print best 100-episode performance

    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))    

    return Q   
# obtain the estimated optimal policy and corresponding action-value function

Q_expsarsa = expected_sarsa(env, 5000, 1)



# print the estimated optimal policy

policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)



print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")

print(policy_expsarsa)



# plot the estimated optimal state-value function

plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])