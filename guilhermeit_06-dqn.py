import gym

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable

from torch.distributions import Categorical

import matplotlib.pyplot as plt

import seaborn as sns
from pyvirtualdisplay import Display

Display().start()



import gym

from IPython import display

import matplotlib.pyplot as plt

%matplotlib inline



env = gym.make('MountainCar-v0')

env.reset()

img = plt.imshow(env.render('rgb_array')) # only call this once



done = False

while not done:

    img.set_data(env.render('rgb_array')) # just update the data

    display.display(plt.gcf())

    display.clear_output(wait=True)

    action = env.action_space.sample()

    next_state, reward, done, info = env.step(action)
!apt-get install python-opengl -y



!apt install xvfb -y



!pip install pyvirtualdisplay



!pip install piglet
# baseado em https://www.kaggle.com/prabodhhere/solving-cartpole-v0-using-reinforce



class Policy(nn.Module):

    """

    Define a politica para tomar uma ação a partir de um estado

    """

    

    

    def __init__(self):

        super(Policy, self).__init__()

        self.num_actions = env.action_space.n

        self.state_dim = env.observation_space.shape[0]

        valN = 16

        self.fc1 = nn.Linear(self.state_dim, valN)

        self.fc2 = nn.Linear(valN, valN)

        self.fc3 = nn.Linear(valN, self.num_actions)

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    

policy = Policy()

    

optimizer = optim.Adam(policy.parameters(), lr=0.001)
def update_policy(states, actions, rewards, log_probs, gamma=0.99):

    """

    Atualiza a rede de politica com gradiente a partir do historico de acoes e recompensas

    """

    loss = []

    dis_rewards = rewards[:]

    for i in range(len(dis_rewards)-2, -1, -1):

        dis_rewards[i] = dis_rewards[i] + gamma * dis_rewards[i+1]

        

    dis_rewards = torch.tensor(dis_rewards)

    for log_prob, reward in zip(log_probs, dis_rewards):

        loss.append(-log_prob * reward)

    

    loss = torch.cat(loss).sum()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    

def get_policy_values(state):

    """

    Calcula o valor de política (ação) a partir do estado

    """

    state = Variable(torch.from_numpy(state)).type(torch.FloatTensor).unsqueeze(0)

    policy_values = policy(state)

    return policy_values
def generate_episode(t_max=1000):

    """

    Gera um episodio e salva estados, ações, recompensas e log prob para atualizar política

    Entrada: passos máximos no episódio

    """

    states, actions, rewards, log_probs = [], [], [], []

    s = env.reset()

    

    for t in range(t_max):

        action_probs = F.softmax(get_policy_values(s), dim=-1)

        sampler = Categorical(action_probs)

        a = sampler.sample()

        log_prob = sampler.log_prob(a)

        new_s, r, done, _ = env.step(a.item())



        states.append(s)

        actions.append(a)

        rewards.append(r)

        log_probs.append(log_prob)

        

        s = new_s

        if done:

            break

    update_policy(states, actions, rewards, log_probs)

    return sum(rewards)
def play_episodes():

    s = env.reset()

    img = plt.imshow(env.render('rgb_array')) # only call this once



    done = False

    while not done:

        img.set_data(env.render('rgb_array')) # just update the data

        display.display(plt.gcf())

        display.clear_output(wait=True)



        action_probs = F.softmax(get_policy_values(s), dim=-1)

        sampler = Categorical(action_probs)

        a = sampler.sample()

        log_prob = sampler.log_prob(a)

        new_s, r, done, _ = env.step(a.item())

        s = new_s
def plot_rewards(rewards, running_rewards):

    """

    Mostra recompensa média (últimos 100) no decorrer da execução

    """

    plt.style.use('seaborn-darkgrid')

    fig = plt.figure(figsize=(12,7))

    ax1 = fig.add_subplot(2, 1, 1)

    ax2 = fig.add_subplot(2, 1, 2)

    plt.subplots_adjust(hspace=.5)

    

    ax1.set_title('Episodic rewards')

    ax1.plot(rewards, label='Episodic rewards')

    ax1.set_xlabel("Episodes")

    ax1.set_ylabel("Rewards")

    

    ax2.set_title('Running rewards')

    ax2.plot(running_rewards, label='Running rewards')

    ax2.set_xlabel("Episodes")

    ax2.set_ylabel("Average rewards")

    

    plt.show(fig)
num_episodes = 1500

verbose = True

print_every = 50

target_avg_reward_100ep = 10

running_reward = None

rewards = []

running_rewards = []

restore_model = True

    

policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=0.001)



# Passa por uma quantidade de episodios e atualiza a politica 



for i in range(num_episodes):

    reward = generate_episode()

    rewards.append(reward)   

    running_reward = np.mean(rewards[-100:])

    running_rewards.append(running_reward)

    

    if verbose:

        if not i % print_every:

            print("Episode: {}. Running reward: {}".format(i+1, running_reward))



    if i >= 99 and running_reward >= target_avg_reward_100ep:

        print("Episode: {}. Running reward: {}".format(i+1, running_reward))

        print("Ran {} episodes. Solved after {} episodes.".format(i+1, i-100+1))

        break

    elif i == num_episodes-1:

        print("Couldn't solve after {} episodes".format(num_episodes))
plot_rewards(rewards, running_rewards)
play_episodes()