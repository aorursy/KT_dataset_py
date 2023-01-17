from abc import ABCMeta, abstractmethod, abstractproperty

import enum



import numpy as np

np.set_printoptions(precision=3)

np.set_printoptions(suppress=True)



import pandas



from matplotlib import pyplot as plt

%matplotlib inline
class BernoulliBandit:

    def __init__(self, n_actions=5):

        self._probs = np.random.random(n_actions)

        

    @property

    def action_count(self):

        return len(self._probs)

    

    def pull(self, action):

        if np.any(np.random.random() > self._probs[action]):

            return 0.0

        return 1.0

    

    def optimal_reward(self):

        """ Used for regret calculation

        """

        return np.max(self._probs)

    

    def step(self):

        """ Used in nonstationary version

        """

        pass

    

    def reset(self):

        """ Used in nonstationary version

        """
class AbstractAgent(metaclass=ABCMeta):   

    def init_actions(self, n_actions):

        self._successes = np.zeros(n_actions)

        self._failures = np.zeros(n_actions)

        self._total_pulls = 0

    

    @abstractmethod

    def get_action(self):

        """

        Get current best action

        :rtype: int

        """

        pass

    

    def update(self, action, reward):

        """

        Observe reward from action and update agent's internal parameters

        :type action: int

        :type reward: int

        """

        self._total_pulls += 1

        if reward == 1:

            self._successes[action] += 1

        else:

            self._failures[action] += 1

    

    @property

    def name(self):

        return self.__class__.__name__





class RandomAgent(AbstractAgent):    

    def get_action(self):

        return np.random.randint(0, len(self._successes))
class EpsilonGreedyAgent(AbstractAgent):

    def __init__(self, epsilon = 0.01):

        self._epsilon = epsilon

    def get_action(self):

        return np.random.randint(len(self._successes)) if np.random.random() < self._epsilon else np.argmax(self._successes/(self._successes+self._failures))

    @property

    def name(self):

        return self.__class__.__name__ + "(epsilon={})".format(self._epsilon)
import math

class UCBAgent(AbstractAgent):

    def get_action(self):

        n_actions = self._successes+self._failures

        ucb = np.sqrt(2*np.log10(self._total_pulls)/n_actions)

        p = self._successes/(n_actions) + ucb

        

        return np.argmax(p)

    @property

    def name(self):

        return self.__class__.__name__
class ThompsonSamplingAgent(AbstractAgent):

    def get_action(self):

        theta = np.array([np.random.beta(self._successes[i], self._failures[i]) if self._successes[i]!=0 and self._failures[i]!=0 else np.random.random() for i in range(len(self._successes))])

        return np.argmax(theta)

    @property

    def name(self):

        return self.__class__.__name__
from collections import OrderedDict



def get_regret(env, agents, n_steps=5000, n_trials=50):

    scores = OrderedDict({

        agent.name: [0.0 for step in range(n_steps)] for agent in agents

    })



    for trial in range(n_trials):

        env.reset()



        for a in agents:

            a.init_actions(env.action_count)



        for i in range(n_steps):

            optimal_reward = env.optimal_reward()



            for agent in agents:

                action = agent.get_action()

                reward = env.pull(action)

                agent.update(action, reward)

                scores[agent.name][i] += optimal_reward - reward



            env.step()  # change bandit's state if it is unstationary



    for agent in agents:

        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials



    return scores



def plot_regret(agents, scores):

    for agent in agents:

        plt.plot(scores[agent.name])



    plt.legend([agent.name for agent in agents])



    plt.ylabel("regret")

    plt.xlabel("steps")



    plt.show()
# Uncomment agents

agents = [

    EpsilonGreedyAgent(),

    UCBAgent(),

    ThompsonSamplingAgent()

]



regret = get_regret(BernoulliBandit(), agents, n_steps=10000, n_trials=10)

plot_regret(agents, regret)