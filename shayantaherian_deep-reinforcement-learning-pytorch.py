!pip install 'kaggle-environments>=0.1.6'
import numpy as np
import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from random import choice
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make
import math
from collections import namedtuple
from itertools import count
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
env = make("connectx", debug=True)


# parameters
BATCH_SIZE = 32
GAMMA = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 2000
class DQN(nn.Module):

    def __init__(self, rows, columns, inarow, outputs):
        super(DQN, self).__init__()
        
        self.rows = rows
        self.columns = columns
        self.inarow = inarow
        
        # Set kernel size to minimum match length
        self.conv1 = nn.Conv2d(1, 16, kernel_size=inarow, stride = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride = 1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input array size, so compute it.
        def conv2d_size_out(size, kernel_size=inarow, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convh = conv2d_size_out(rows)
        convw = conv2d_size_out(columns + inarow - 1)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.reshape(x, (-1, 1, self.rows, self.columns))
        x = torch.cat((x, torch.zeros(x.shape[0], 1, self.rows, self.inarow-1)), dim=3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = env.configuration['columns']
rows = env.configuration['rows']

# Number of actions is equal to number of columns
n_actions = columns
inarow = env.configuration['inarow']
policy_net = DQN(rows, columns, inarow, n_actions).to(device)
def select_action(observation):
    steps_done = 0
    state = torch.tensor(observation.board, dtype=torch.float)
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        np.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    def __len__(self):
        return len(self.memory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
#agent = Agent(strategy, n_actions, device)
memory = ReplayMemory(memory_size)
policy_net = DQN(rows, columns, inarow, n_actions).to(device)
target_net = DQN(rows, columns, inarow, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

def plot_duration(values, moving_avg_period):
    plt.figure(figsize = (14,6))
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(tuple(ten.type(torch.long) for ten in batch.action))
    reward_batch = torch.tensor(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward'))

trainer = env.train([None, "negamax"])
episode_durations = []
for episode in range(num_episodes):
    rewards = 0
    iter = 0
    done = False
    #losses = []
    observation = trainer.reset()
    for timestep in count():
        state = torch.tensor(observation.board, dtype=torch.float)
        #action = agent.select_action(observation,policy_net)
        action = select_action(observation)
        chosen_column = action.item()
        last_state = state
        observation, reward, done, info = trainer.step(chosen_column)
        if not done:
            next_state = torch.tensor(observation.board, dtype = torch.float)
        else:
            next_state = None
        
        if done:
            if reward == 1: # won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else: #Draw
                reward = 10
        elif reward == None:
            reward = 0
        
        memory.push(Experience(last_state, action, next_state, reward))
        optimize_model()
        #losses.append(loss)
            
        if done:
            episode_durations.append(timestep)
            plot_duration(episode_durations, 100)
            break
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
          
 
    
torch.set_printoptions(profile="full")

agent = f"""
import torch
import numpy as np
import random
from torch import nn, tensor
import torch.nn.functional as F
from collections import OrderedDict
    
def my_agent(observation, configuration):
    class DQN(nn.Module):
        def __init__(self, rows, columns, inarow, outputs):
            super(DQN, self).__init__()

            self.rows = rows
            self.columns = columns
            self.inarow = inarow

            self.conv1 = nn.Conv2d(1, 16, kernel_size=inarow, stride = 1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride = 1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride = 1)
            self.bn3 = nn.BatchNorm2d(32)

            def conv2d_size_out(size, kernel_size=inarow, stride = 1):
                return (size - (kernel_size - 1) - 1) // stride  + 1
            convh = conv2d_size_out(rows)
            convw = conv2d_size_out(columns + inarow - 1)
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)

        def forward(self, x):
            x = torch.reshape(x, (-1, 1, self.rows, self.columns))
            x = torch.cat((x, torch.zeros(x.shape[0], 1, self.rows, self.inarow-1)), dim=3)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
            
    columns = configuration['columns']
    rows = configuration['rows']

    n_actions = columns
    inarow = configuration['inarow']
    
    policy_net = DQN(rows, columns, inarow, n_actions)
    
    policy_net.load_state_dict({str(policy_net.state_dict())})
    policy_net.eval()
        
    state = torch.tensor(observation.board, dtype=torch.float)
    
    with torch.no_grad():
        action = policy_net(state).max(1)[1].view(1, 1)
        
    if observation.board[action] != 0:
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])
        
    return int(action[0][0].item())
"""
with open('submission.py', 'w') as f:
    f.write(agent)
from submission import my_agent
env.play([None, my_agent], width=500, height=450)
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=my_agent, agent2="random")