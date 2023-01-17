import numpy as np

from collections import defaultdict



class Agent:



    def __init__(self, nA=6, alpha=.01, gamma=1, epsilon=0):

        """ Initialize agent.



        Params

        ======

        - nA: number of actions available to the agent

        """

        self.nA = nA

        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.alpha = alpha

        self.gamma = gamma

        self.epsilon = epsilon

    

    def get_best_action(self, Q_state):

        # if there is more than one best, return any of them randomly

        indices_with_max_Q = np.argwhere( np.max( Q_state ) == Q_state) 

        best_action = np.random.choice( indices_with_max_Q.reshape(-1) ) 

        return best_action



    def select_action(self, state):

        """ Given the state, select an action.



        Params

        ======

        - state: the current state of the environment



        Returns

        =======

        - action: an integer, compatible with the task's action space

        """

        if np.random.rand(1) < self.epsilon:

            # chooses randomly

            action = np.random.choice( np.arange( self.nA ) )

        else:

            # chooses maximum policy

            action = self.get_best_action(self.Q[state])



        return action



    def step(self, state, action, reward, next_state, done):

        """ Update the agent's knowledge, using the most recently sampled tuple.



        Params

        ======

        - state: the previous state of the environment

        - action: the agent's previous choice of action

        - reward: last reward received

        - next_state: the current state of the environment

        - done: whether the episode is complete (True or False)

        """

        next_q = self.Q[next_state][self.get_best_action(self.Q[next_state])]

        self.Q[state][action] += self.alpha * (reward + self.gamma * next_q - self.Q[state][action])
from collections import deque

import sys

import math

import numpy as np

import time



def interact(env, agent, num_episodes=20000, window=100):

    """ Monitor agent's performance.

    

    Params

    ======

    - env: instance of OpenAI Gym's Taxi-v1 environment

    - agent: instance of class Agent (see Agent.py for details)

    - num_episodes: number of episodes of agent-environment interaction

    - window: number of episodes to consider when calculating average rewards



    Returns

    =======

    - avg_rewards: deque containing average rewards

    - best_avg_reward: largest value in the avg_rewards deque

    """

    # initialize average rewards

    avg_rewards = deque(maxlen=num_episodes)

    # initialize best average reward

    best_avg_reward = -math.inf

    # initialize monitor for most recent rewards

    samp_rewards = deque(maxlen=window)

    # for each episode

    disp = True

    for i_episode in range(1, num_episodes+1):

        # begin the episode

        state = env.reset()

        # initialize the sampled reward

        samp_reward = 0

        while True:

            if disp:

                display.clear_output(wait=True)

                env.render()

                time.sleep(0.05)

            # agent selects an action

            action = agent.select_action(state)

            # agent performs the selected action

            next_state, reward, done, _ = env.step(action)

            # agent performs internal updates based on sampled experience

            agent.step(state, action, reward, next_state, done)

            # update the sampled reward

            samp_reward += reward

            # update the state (s <- s') to next time step

            state = next_state

            if done:

                # save final sampled reward

                samp_rewards.append(samp_reward)

                break

        disp = False

        if (i_episode % 100 == 0): disp=True

        if (i_episode >= 100):

            # get average reward from last 100 episodes

            avg_reward = np.mean(samp_rewards)

            # append to deque

            avg_rewards.append(avg_reward)

            # update best average reward

            if avg_reward > best_avg_reward:

                best_avg_reward = avg_reward

        # monitor progress

        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")

        sys.stdout.flush()

        # check if task is solved (according to OpenAI Gym)

        if best_avg_reward >= 9.7:

            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")

            break

        if i_episode == num_episodes: print('\n')

    return avg_rewards, best_avg_reward
# from agent import Agent

# from monitor import interact

import gym

import numpy as np



env = gym.make('Taxi-v2')

agent = Agent()

avg_rewards, best_avg_reward = interact(env, agent, num_episodes=10000)