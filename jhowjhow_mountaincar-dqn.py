!apt-get install python-opengl -y



!apt install xvfb -y



!pip install pyvirtualdisplay



!pip install piglet
import gym

import numpy as np

import random

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable

from torch.distributions import Categorical

import matplotlib.pyplot as plt

import seaborn as sns
num_episodes = 1000

verbose = True

print_every = 100

target_avg_reward_100ep = 10

rewards = []

running_rewards = []

restore_model = True

successes = 0



gamma=0.95

epsilon=.2
from pyvirtualdisplay import Display

Display().start()



import gym

from IPython import display

import matplotlib.pyplot as plt

%matplotlib inline



env = gym.make('MountainCar-v0')

env.reset()

# img = plt.imshow(env.render('rgb_array'))
class Policy(nn.Module):

    """

    Define a politica para tomar uma ação a partir de um estado

    """

    def __init__(self):

        super(Policy, self).__init__()

        self.num_actions = env.action_space.n

        self.state_dim = env.observation_space.shape[0]

        self.fc1 = torch.nn.Linear(self.state_dim,64,'relu')

        self.fc2 = torch.nn.Linear(64,self.num_actions,'linear')

        

    def forward(self,x):

        x = F.relu(self.fc1(x))

        y = self.fc2(x)

        return y
# Usado como base: https://github.com/orrivlin/MountainCar_DQN_RND

class Policy2(nn.Module):

    """

    Define a politica para tomar uma ação a partir de um estado

    """

    def __init__(self):

        super(Policy2, self).__init__()

        self.num_actions = env.action_space.n

        self.state_dim = env.observation_space.shape[0]    

        self.fc1 = torch.nn.Linear(self.state_dim,64,'linear')

        self.hidden1 = nn.Dropout(0.1)

        self.fc2 = torch.nn.Linear(64,64,'linear')

        self.hidden2 = nn.Dropout(0.08)

        self.fc3 = torch.nn.Linear(64,32,'linear')

        self.hidden3 = nn.Dropout(0.05)

        self.fc4 = torch.nn.Linear(32,16,'linear')

        self.fc5 = torch.nn.Linear(16,self.num_actions,'linear')

        

    def forward(self,x):

        

        x = self.hidden1(self.fc1(x))

        x = self.hidden2(F.relu(self.fc2(x)))

        x = self.hidden3(F.relu(self.fc3(x)))        

        x = F.relu(self.fc4(x))

        y = F.relu(self.fc5(x))

        return y
def get_policy_values(state,policy):

    """

    Calcula o valor de política (ação) a partir do estado

    """

    state = Variable(torch.from_numpy(state)).type(torch.FloatTensor).unsqueeze(0)

    policy_values = policy(state)

    return policy_values
# Utliziado como base: https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2

def generate_episode(policy,loss_fn, optimizer, successes, epsilon, gamma, t_max=1000):

    """

    Gera um episodio e salva estados, ações, recompensas e log prob para atualizar política

    Entrada: passos máximos no episódio

    """

    states, actions, rewards, log_probs = [], [], [], []

    state = env.reset()

    

    for t in range(t_max):

        Q = get_policy_values(state,policy)

        

        if np.random.rand() <= epsilon:

            action = random.randrange(policy.num_actions)

        else:

            _, action = torch.max(Q, -1)

            action = action.item()

            

            

       # Step forward and receive next state and reward

        state_1, reward, done, _ = env.step(action)

        

        # Adjust reward based on car position

        reward = state_1[0] + 0.5

        

        # Adjust reward for task completion

        if state_1[0] >= 0.5:

            reward += 1

            print("Solved ",done)

        

        # Find max Q for t+1 state

        Q1 = get_policy_values(state_1,policy)

        maxQ1, _ = torch.max(Q1, -1)

        

        # Create target Q value for training the policy

        Q_target = Q.clone()

        Q_target = Variable(Q_target)

        Q_target[0][action] = reward + torch.mul(maxQ1.detach(), gamma)

        

        # Calculate loss

        loss = loss_fn(Q, Q_target)

        

        # Update policy

        policy.zero_grad()

        loss.backward()

        optimizer.step()

        

        rewards.append(reward)



        if done:

            if state_1[0] >= 0.5:

                epsilon *= .9

                successes += 1

            break

        else:

            state = state_1

    return policy, loss_fn, rewards, epsilon, successes
def play_episodes(policy):

    state = env.reset()

    img = plt.imshow(env.render('rgb_array')) # only call this once



    done = False

    while not done:

        img.set_data(env.render('rgb_array')) # just update the data

        display.display(plt.gcf())

        display.clear_output(wait=True)

        

        Q = get_policy_values(state,policy)

        _, action = torch.max(Q, -1)

        state,reward,done,_ = env.step(action.item())
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
def test(policy, episode, t_max=1000):

    if episode % 100 == 0 and episode > 10:

        total_reward = []

        for i in range(10):

            state = env.reset()

            for j in range(t_max):

                Q = get_policy_values(state,policy)

                _, action = torch.max(Q, -1)

                state,reward,done,_ = env.step(action.item())

                  

                # Adjust reward based on car position

                reward = state[0] + 0.5



                # Adjust reward for task completion

                if state[0] >= 0.5:

                    reward += 1

                    print("Solved ",done)

                total_reward.append(reward)

                if done:

                    break

        ave_reward = np.mean(total_reward)

        print("Test: Episode: {}. Running reward: {}".format(episode, ave_reward))

        return ave_reward > .3

    return False
def train(policy,loss_fn, optimizer, num_episodes = 1000):

    rewards = []

    running_rewards = []

    successes = 0

    epsilon = .1

    gamma=0.9

    for i in range(num_episodes):

        

        state = env.reset()

        

        policy,loss_fn,reward,epsilon,successes = generate_episode(policy,loss_fn,optimizer, successes, epsilon, gamma) 

        rewards.append(sum(reward))   

        running_reward = np.mean(reward)

        running_rewards.append(running_reward)

        

        if verbose:

            if not i % print_every:

                print("Episode: {}. Running reward: {}. Epsilon: {}".format(i+1, running_reward, epsilon))



        if test(policy, i):

            print("Ran {} episodes. Solved after {} episodes.".format(i+1, i-100+1))

            break

            

        if i == num_episodes-1 and not successes:

            print("Couldn't solve after {} episodes".format(num_episodes))

            

    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/num_episodes*100))

    return rewards, running_rewards
verbose = True

print_every = 100

running_reward = None

rewards = []

running_rewards = []

restore_model = True
print("shallow learning")

state = env.reset()

policy = Policy()

loss_fn = nn.MSELoss()

optimizer = optim.Adam(policy.parameters(), lr=0.001)

rewards,running_rewards = train(policy, loss_fn, optimizer, num_episodes=1000)
plot_rewards(rewards, running_rewards)
play_episodes(policy)
verbose = True

print_every = 100

running_reward = None

rewards = []

running_rewards = []

restore_model = True
print("shallow learning 2")

state = env.reset()

policy2 = Policy2()

loss_fn = nn.MSELoss()

optimizer2 = optim.Adam(policy2.parameters(), lr=0.001)

rewards,running_rewards = train(policy2, loss_fn, optimizer2,num_episodes=4000)
plot_rewards(rewards, running_rewards)
play_episodes(policy2)