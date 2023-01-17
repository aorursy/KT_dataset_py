import os

import torch

import numpy as np

from torch import nn

from time import time

from torch import optim

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from kaggle_environments import evaluate, make
start_time = time()
class ConnectX:

    def __init__(self, switch_prob=0.5):

        self.env = make('connectx', debug=False)

        self.pair = [None, 'negamax']

        self.trainer = self.env.train(self.pair)

        self.switch_prob = switch_prob

        self.config = self.env.configuration

        self.action_space = self.config.columns

        self.state_space = self.config.columns * self.config.rows

        self.rule = {True: {1: 20.0, 0: -20.0, None: 10.0}, False: {0.5: 5.0}}



    def switch_trainer(self):

        self.pair[1] = np.random.choice(['random','negamax'])

        self.trainer = self.env.train(self.pair)



    def step(self, action):

        observations, reward, done, _ = self.trainer.step(action)

        reward = self.rule[done][reward]

        return observations.board, reward, done



    def reset(self):

        if np.random.random() < self.switch_prob:

            self.switch_trainer()

        observations = self.trainer.reset()

        return observations.board



    def render(self, **kwargs):

        return self.env.render(**kwargs)
class Experience:

    class Memory:

        def __init__(self, curr_state, action, reward, done, next_state):

            self.curr_state = curr_state

            self.action = action

            self.reward = reward

            self.done = done

            self.next_state = next_state



    def __init__(self, memory_size):

        self.memory_size = memory_size

        self.memories = []



    def choice(self, size):

        return np.random.choice(self.memories, min(len(self.memories), size))



    def update(self, curr_state, action, reward, done, next_state):

        # if memory is full, remove the oldest transition

        self.memories = self.memories[1:] if len(self.memories) >= self.memory_size else self.memories

        self.memories.append(self.Memory(curr_state, action, reward, done, next_state))
class DQN(nn.Module):

    def __init__(self, num_states, num_actions, hidden_units = 512):

        super(DQN, self).__init__()

        self.num_states = num_states

        self.hidden_units = hidden_units

        self.num_actions = num_actions

        self.experience = Experience(10000)

        

        self.layer1 = nn.Sequential(

            nn.Linear(num_states, hidden_units),

            nn.LayerNorm(hidden_units),

            nn.ReLU())

        

        self.layer2 = nn.Sequential(

            nn.Linear(hidden_units, num_actions),

            nn.Sigmoid())



    def forward(self, states):

        states = states.view(-1, self.num_states)

        output = self.layer1(states)

        actions = self.layer2(output)

        return actions
class Trainer:

    def __init__(self, num_episodes, batch_size = 512):

        self.num_episodes = num_episodes

        self.gamma = 0.99

        self.final_epsilon = 0.0001

        self.init_epsilon = 0.1

        self.epsilon_decay = (self.init_epsilon - self.final_epsilon) / (self.num_episodes - 1)

        self.batch_size = batch_size

        self.device = torch.device(type = 'cuda' if torch.cuda.is_available() else 'cpu')

        self.env = ConnectX()

        self.model = DQN(self.env.state_space, self.env.action_space).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)

        self.criterion = nn.MSELoss()

        self.rewards = np.array([])



    def epsilon_decrements(self, episode_no):

        epsilon = self.init_epsilon - self.epsilon_decay * episode_no

        return epsilon



    def train(self):

        # sample random minibatch

        experiences = self.model.experience.choice(self.batch_size)

        # unpack minibatch

        curr_state_batch = torch.tensor([experience.curr_state for experience in experiences], device = self.device, dtype = torch.float)

        action_batch = torch.tensor([experience.action for experience in experiences], device = self.device)

        reward_batch = torch.tensor([experience.reward for experience in experiences], device = self.device)

        done_batch = torch.tensor([experience.done for experience in experiences], device = self.device)

        next_state_batch = torch.tensor([experience.next_state for experience in experiences], device = self.device, dtype = torch.float)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)

        y_batch = torch.where(done_batch, reward_batch, self.gamma * self.model(next_state_batch).argmax(1))

        # extract Q-value

        q_value = self.model(curr_state_batch)[:,action_batch].sum(dim = 1)

        return q_value, y_batch

    

    def play(self, episode_no):

        curr_state = self.env.reset()

        done = False

        total_reward = 0

        while not done:

            # epsilon annealing

            epsilon = self.epsilon_decrements(episode_no)

            # epsilon greedy exploration

            random_action = np.random.uniform() <= epsilon

            # get output from the neural network or random

            if random_action:

                action = torch.randint(self.model.num_actions, torch.Size((1,)))  

            else:

                action = self.model(torch.tensor(curr_state, device = self.device, dtype = torch.float)).argmax(1)

            # get next state and reward

            next_state, reward, done = self.env.step(action.item())

            # update experience

            self.model.experience.update(curr_state, action, reward, done, next_state)

            # Optize model

            q_value, y_batch = self.train()

            # calculate loss

            loss = self.criterion(q_value, y_batch)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass

            self.optimizer.zero_grad()

            # do backward pass

            loss.backward()

            self.optimizer.step()

            # set curr_state to be next_state

            curr_state = next_state

            total_reward += reward

        return total_reward



    def __call__(self):

        for n in tqdm(range(len(self.rewards), self.num_episodes)):

            self.rewards = np.append(self.rewards, self.play(n))

            if (time() - start_time) / 3600 > 2:

                break



    def agent(self, observation, config):

        state = torch.tensor(observation.board, device = self.device, dtype = torch.float)

        actions = self.model(state).topk(config.columns, dim = 1)[1][0]

        actions = [action.item() for action in actions if observation.board[action.item()] == 0]

        return actions[0]



    def load_state_dict(self, path):

        checkpoint = torch.load(path) if os.path.exists(path) else {}

        self.model.load_state_dict(checkpoint.get('model', self.model.state_dict()))

        self.rewards = checkpoint.get('rewards', self.rewards)
trainer = Trainer(50000)

trainer.load_state_dict('../input/connectx-using-deep-q-learning/model.pth')
trainer()
avg_rewards = np.array([]) # Last 100 steps

for n in range(len(trainer.rewards)):

    avg_rewards = np.append(avg_rewards, trainer.rewards[max(0, n - 100):(n + 1)].mean())

plt.figure(figsize = (15,5))

plt.plot(avg_rewards)

plt.xlabel('Episode')

plt.ylabel('Avg rewards (100)')

plt.show()
torch.save({'model': trainer.model.state_dict(), 'rewards': trainer.rewards}, 'model.pth')
def my_agent(observation, configuration):

    return trainer.agent(observation, configuration)
env = make('connectx', debug = True)

# Test agent

env.reset()

# Play as the first agent against default "random" agent.

env.run([my_agent, "negamax"])

env.render(mode="ipython", width=500, height=450)
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)
# Run multiple episodes to estimate its performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
state_dict = dict([(key, value.cpu().numpy().tolist()) for key, value in trainer.model.state_dict().items()])

state_dict = str(state_dict)
subission = f'''

import torch

import numpy as np

from torch import nn

from torch import optim

from collections import OrderedDict

from kaggle_environments import make



env = make("connectx", debug = False)

device = torch.device(type = 'cuda' if torch.cuda.is_available() else 'cpu')



state_dict = {state_dict}

state_dict = OrderedDict((key, torch.tensor(value, device = device)) for key, value in state_dict.items())



class DQN(nn.Module):

    def __init__(self, num_states, num_actions, hidden_units = 512):

        super(DQN, self).__init__()

        self.num_states = num_states

        self.hidden_units = hidden_units

        self.num_actions = num_actions

        

        self.layer1 = nn.Sequential(

            nn.Linear(num_states, hidden_units),

            nn.LayerNorm(hidden_units),

            nn.ReLU())

        

        self.layer2 = nn.Sequential(

            nn.Linear(hidden_units, num_actions),

            nn.Sigmoid())



    def forward(self, states):

        states = states.view(-1, self.num_states)

        output = self.layer1(states)

        actions = self.layer2(output)

        return actions



model = DQN(env.configuration.columns * env.configuration.rows, env.configuration.columns).to(device)

model.load_state_dict(state_dict)



def my_agent(observation, configuration):

    state = torch.tensor(observation.board, device = device, dtype = torch.float)

    actions = model(state).topk(configuration.columns, dim = 1)[1][0]

    actions = [action.item() for action in actions if observation.board[action.item()] == 0]

    return actions[0]

'''
with open('submission.py', 'w') as f:

    f.write(subission)

"%s Kb" % round(os.stat('submission.py').st_size/1024)
import sys

sys.path.append('../input/connectx-using-deep-q-learning')