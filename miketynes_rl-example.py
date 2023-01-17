import gym

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical

import torchvision.transforms as T



from tqdm import tqdm



from matplotlib import pyplot as plt



import warnings

warnings.simplefilter('ignore')
env = gym.make('CartPole-v1')
# Policy: a simple 2 layer network!



class Policy(nn.Module):

    def __init__(self):

        super(Policy, self).__init__()

        self.sequence = nn.Sequential(

            nn.Linear(4, 10),

            nn.ReLU(),

            nn.Linear(10, 2),

            nn.Softmax(dim=1)

        )



    def forward(self, x):

        return self.sequence(x)





# we also have a critic nework (don't worry about this)

class Linear(nn.Module):

    def __init__(self):

        super(Linear, self).__init__()

        self.linear = nn.Linear(4, 1)



    def forward(self, x):

        return self.linear(x)
GAMMA = 0.99



total_rewards = []



# initialize policy 'actor' network

policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=.002)

eps = np.finfo(np.float32).eps.item()



# second 'critic' network (we wont worry about this)

v = Linear()

v_loss = nn.MSELoss()

v_optimizer = optim.SGD(v.parameters(), lr=0.01)





# for each 'episode', i.e. 'playthrougg'

for i in tqdm(range(500)):

    state = env.reset()



    rewards = []

    states = [state]

    log_probs = []



    terminal = False



    # for each state until end state reached

    while not terminal:

        

        # Select an action

        state = torch.from_numpy(state).float().unsqueeze(0)

        c = Categorical(policy(state))

        action = c.sample()

        log_probs.append(c.log_prob(action))



        # feed the action to the environment, 

        # get next state, reward, if done, _

        state, reward, terminal, _ = env.step(action.item())



        if not terminal:

            states.append(state)



        rewards.append(reward)



    total_rewards.append(sum(rewards))



    # Calculate the returns

    R = 0

    returns = []

    for r in reversed(rewards):

        R = r + GAMMA * R

        returns.insert(0, R)

    returns = torch.tensor(returns)



    values = v(torch.FloatTensor(states))



    # Calculate the loss

    policy_loss = [-log_prob * R for log_prob, R in zip(log_probs, returns)]



    # Backprop for the NN

    optimizer.zero_grad()

    policy_loss = torch.cat(policy_loss).sum()

    policy_loss.backward()

    optimizer.step()



    # Backprop for the value estimator

    v_optimizer.zero_grad()

    estimate_loss = v_loss(values, returns)

    estimate_loss.backward()

    v_optimizer.step()
N=20

plt.figure(figsize=(15,10))

plt.plot(np.convolve(total_rewards, np.ones((N,))/N, mode='valid'))

plt.xlabel('Episode')

plt.ylabel('Cumulative reward')

plt.show()