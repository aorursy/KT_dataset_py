#!pip3 uninstall tensorforce
!pip3 install tensorforce

from numba import cuda
cuda.select_device(0)
cuda.close()
# Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self, action):
        raise NotImplementedError('Inheriting classes must override step')

import random


def new_arrival(arrival_probability):
    """
    Returns true if there is a new arrival.
    :param arrival_probability: Arrival Probability.
    :return: Whether there is a new arrival.
    """
    rand_int = random.randint(0, 100)  # produces 0 to range
    return True if rand_int <= arrival_probability else False
class BernoulliQueue(Environment):
    def __init__(self, flow_id, arrival_probability, is_fifo, is_debug):
        self.flow_id = flow_id
        self.queue_list = []
        self.arrival_probability = arrival_probability
        self.is_fifo = is_fifo
        self.current_aoi = 0
        self.time_step = 0
        self.avg_aoi = 0

        self.is_debug = is_debug

    def reset(self):
        self.queue_list.clear()
        self.queue_list = []
        self.current_aoi = 0
        self.avg_aoi = 0
        self.time_step = 0

    def actions(self):
        if len(self.queue_list) > 0:
            if self.is_fifo:
                return self.__front()
            else:
                return self.__rear()
        else:
            return -1

    def __front(self):
        return self.queue_list[len(self.queue_list) - 1]

    def __rear(self):
        return self.queue_list[0]

    def step(self, action):
        popped_item = -1
        self.current_aoi += 1
        self.time_step += 1

        if action:
            if self.is_fifo:
                popped_item = self.queue_list.pop()
            else:
                popped_item = self.queue_list.pop(0)

        for x in range(0, len(self.queue_list)):
            self.queue_list[x] += 1

        is_new_arrival = new_arrival(self.arrival_probability)
        if is_new_arrival:
            self.queue_list.insert(0, 1)

        if popped_item != -1 and popped_item < self.current_aoi:
            self.current_aoi = popped_item

        # calculate avg aoi
        self.avg_aoi = (self.avg_aoi * (self.time_step - 1) + self.current_aoi) / self.time_step

        if self.is_debug:
            print("flow_id: ", self.flow_id, " time_step: ", self.time_step, "queue: ", self.queue_list,
                  " current_aoi: ", self.current_aoi,
                  " avg_aoi: ", "{0:.2f}".format(self.avg_aoi), " action: ", action)

        return popped_item, self.current_aoi, self.avg_aoi, is_new_arrival

from tensorforce import Environment
import numpy as np

class MultiQueueSystem(Environment):
    def __init__(self):
        super(MultiQueueSystem, self).__init__()

        self.number_of_flows = 4
        self.arrival_probability = 20
        self.is_fifo = True
        self.flows = []
        self.time_step = 0
        self.avg_aoi_all = 0

        for i in range(self.number_of_flows):
            self.flows.append(BernoulliQueue(i + 1, self.arrival_probability, self.is_fifo, False))

    def states(self):
        return dict(type='int', shape=(self.number_of_flows,), num_values=1000000)

    def actions(self):
        # actions = [0]
        # for i in range(self.number_of_flows):
        #     if self.flows[i].actions() > 0:
        #         actions.append(i + 1)
        # return np.array(actions)
        return dict(type='int', num_values=self.number_of_flows + 1)

    # def states(self):
    #     current_state = []
    #     for i in range(self.number_of_flows):
    #         current_state.append(self.flows[i].current_aoi)  # current state

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def reset(self):
        self.time_step = 0
        self.avg_aoi_all = 0

        for i in range(self.number_of_flows):
          self.flows[i].reset()

        return dict(
            state=np.zeros(shape=(self.number_of_flows,), dtype=np.int),
            action_mask=self.__calculate_action_mask()
        )

    def available_actions(self):
        #buna bi emin ol
        actions = [0]
        for i in range(self.number_of_flows):
            if self.flows[i].actions() > 0:
                actions.append(i + 1)
        return actions

    def execute(self, actions):
        """
        Progress the environment one time step.

        :param action: Schedule the packet on the flow, idle if 0.
        """
        reward = 0
        action = actions
        self.time_step += 1

        for i in range(self.number_of_flows):
            if action == i + 1:
                if len(self.flows[i].queue_list) > 0:
                    self.flows[i].step(True)
                else:
                    reward = -100
            else:
                self.flows[i].step(False)

        # calculate the total average aoi
        current_avg_aoi = 0
        current_state = []
        for i in range(self.number_of_flows):
            current_avg_aoi += self.flows[i].avg_aoi
            current_state.append(self.flows[i].current_aoi)  # current state
        current_avg_aoi = current_avg_aoi / self.number_of_flows
        self.avg_aoi_all = (self.avg_aoi_all * (self.time_step - 1) + current_avg_aoi) / self.time_step

        # reward
        if reward > -100:
            for aoi in current_state:
                reward += -aoi

        # no natural ending point
        terminal = False
        # if self.max_episode_timesteps() <= self.time_step:
        #     terminal = True

        states = dict(
            state=np.array(current_state),
            action_mask=self.__calculate_action_mask()
        )

        return states, np.array(terminal), np.array(reward)

    def __calculate_action_mask(self):
        # calculate action mask
        actions = [True]
        for i in range(self.number_of_flows):
            if self.flows[i].actions() > 0:
                actions.append(True)
            else:
                actions.append(False)
        return actions

    def get_avg_aoi(self):
        return self.avg_aoi_all


# # test
# random.seed(200)
# state = np.array([0, 0, 0, 0])
# env = MultiQueueSystem()
# for x in range(1000):
#     # MAF scheduling
#     action = 0
#     max_num = 0
#     actions_list = env.actions()
#     for a in actions_list:
#         if a != 0:
#             if max_num < state[a - 1]:
#                 max_num = state[a - 1]
#                 action = a
#
#     # if max(state) > 0:
#     #     action = state.index(max(state)) + 1
#
#     state, terminal, current_reward = env.execute(action)
#     print("time_step: ", x + 1, " current_state: ", state, " avg_aoi_all: ", "{0:.2f}".format(env.avg_aoi_all),
#           " action_taken: ", action, " reward: ", current_reward,
#           "\n")

def maf(env,time_steps):
  env.reset()
  state = np.array([0, 0, 0, 0])
  for x in range(time_steps):
      # MAF scheduling
      action = 0
      max_num = 0
      actions_list = env.available_actions()
      for a in actions_list:
          if a != 0:
              if max_num < state[a - 1]:
                  max_num = state[a - 1]
                  action = a

      if max(state) > 0:
          action = state.argmax() + 1
      states, terminal, current_reward = env.execute(action)
      state = states.get("state")

  return env.get_avg_aoi()
from tensorforce import Agent, Environment

DEBUG = False

# Pre-defined or custom environment
environment = Environment.create(
    environment=MultiQueueSystem, max_episode_timesteps=50
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    agent= 'dqn',
    memory= 100000,
    learning_rate= 0.01,
    network=dict(type='auto', size=10)
)

# Instantiate a Tensorforce agent
agent_dueling_dqn = Agent.create(
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    agent= 'dueling_dqn',
    memory= 100000,
    learning_rate= 0.01,
    network=dict(type='auto', size=10)
)

# Instantiate a Tensorforce agent
agent_ac = Agent.create(
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    agent= 'ac',
    memory= 100000,
    learning_rate= 0.01,
    network=dict(type='auto', size=10)
)

RANGE = 200
avg_aoi_list_maf = []
avg_aoi_list_dqn = []
avg_aoi_list_ddqn = []
avg_aoi_list_ac = []

for _ in range(RANGE):

    seed = random.randint(0, 100)

    random.seed(seed)
    avg_aoi_maf = maf(MultiQueueSystem(),100)
    print("avg_aoi_maf: ",avg_aoi_maf)
    avg_aoi_list_maf.append(avg_aoi_maf)

    # Initialize episode
    states = environment.reset()
    terminal = False

    random.seed(seed)
    i = 1
    avg_aoi = 0
    while not terminal:
        # dqn
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        # calculate avg aoi
        current_sum = np.sum(states.get("state")) / 4
        avg_aoi = (avg_aoi * (i - 1) + current_sum) / i
        if DEBUG:
            print("time_step: ", i, " state: ", states, " reward: ", reward, " action: ", actions, " avg_aoi: ",
                  "{0:.2f}".format(avg_aoi))
        i += 1

    print("avg_aoi_dqn: ", "{0:.2f}".format(avg_aoi))
    avg_aoi_list_dqn.append(avg_aoi)

    # ddqn
    random.seed(seed)

    # Initialize episode
    states = environment.reset()
    terminal = False

    j = 1
    avg_aoi_ddqn = 0
    while not terminal:
        # dqn
        # Episode timestep
        actions = agent_dueling_dqn.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent_dueling_dqn.observe(terminal=terminal, reward=reward)

        # calculate avg aoi
        current_sum = np.sum(states.get("state")) / 4
        avg_aoi_ddqn = (avg_aoi_ddqn * (j - 1) + current_sum) / j
        if DEBUG:
            print("time_step: ", j, " state: ", states, " reward: ", reward, " action: ", actions, " avg_aoi_ddqn: ",
                  "{0:.2f}".format(avg_aoi_ddqn))
        j += 1

    print("avg_aoi_ddqn: ", "{0:.2f}".format(avg_aoi_ddqn))
    avg_aoi_list_ddqn.append(avg_aoi_ddqn)

    # ac
    random.seed(seed)

    # Initialize episode
    states = environment.reset()
    terminal = False

    k = 1
    avg_aoi_ac = 0
    while not terminal:
        # dqn
        # Episode timestep
        actions = agent_ac.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent_ac.observe(terminal=terminal, reward=reward)

        # calculate avg aoi
        current_sum = np.sum(states.get("state")) / 4
        avg_aoi_ac = (avg_aoi_ac * (k - 1) + current_sum) / k
        if DEBUG:
            print("time_step: ", j, " state: ", states, " reward: ", reward, " action: ", actions, " avg_aoi_ddqn: ",
                  "{0:.2f}".format(avg_aoi_ac))
        k += 1

    print("avg_aoi_ac: ", "{0:.2f}".format(avg_aoi_ac),"\n")
    avg_aoi_list_ac.append(avg_aoi_ac)

agent.close()
print("buraya geldim")
environment.close()

 
import matplotlib.pyplot as plt


fig= plt.figure(figsize=(20,10))
plt.plot(range(RANGE), avg_aoi_list_dqn, label = "DQN")
plt.plot(range(RANGE), avg_aoi_list_ddqn, label = "DuelingDQN")
plt.plot(range(RANGE), avg_aoi_list_maf, label = "MAF")
plt.plot(range(RANGE), avg_aoi_list_ac, label = "AC")

plt.xlabel('episodes') 
plt.ylabel('Average AoI (time unit)') 

plt.legend() 
plt.grid(True)
plt.show() 
import matplotlib.pyplot as plt


fig= plt.figure(figsize=(20,10))
#plt.plot(range(RANGE), avg_aoi_list_dqn, label = "DQN")
plt.plot(range(RANGE), avg_aoi_list_ddqn, label = "DuelingDQN")
plt.plot(range(RANGE), avg_aoi_list_maf, label = "MAF")
#plt.plot(range(RANGE), avg_aoi_list_ac, label = "AC")

plt.xlabel('episodes') 
plt.ylabel('Average AoI (time unit)') 

plt.legend() 
plt.grid(True)
plt.show() 
import matplotlib.pyplot as plt


fig= plt.figure(figsize=(20,10))
#plt.plot(range(RANGE), avg_aoi_list_dqn, label = "DQN")
#plt.plot(range(RANGE), avg_aoi_list_ddqn, label = "DuelingDQN")
plt.plot(range(RANGE), avg_aoi_list_maf, label = "MAF")
plt.plot(range(RANGE), avg_aoi_list_ac, label = "AC")

plt.xlabel('episodes') 
plt.ylabel('Average AoI (time unit)') 

plt.legend() 
plt.grid(True)
plt.show() 
import matplotlib.pyplot as plt


fig= plt.figure(figsize=(20,10))
plt.plot(range(RANGE), avg_aoi_list_dqn, label = "DQN")
plt.plot(range(RANGE), avg_aoi_list_ddqn, label = "DuelingDQN")
plt.plot(range(RANGE), avg_aoi_list_maf, label = "MAF")
#plt.plot(range(RANGE), avg_aoi_list_ac, label = "AC")

plt.xlabel('episodes') 
plt.ylabel('Average AoI (time unit)') 

plt.legend() 
plt.grid(True)
plt.show() 