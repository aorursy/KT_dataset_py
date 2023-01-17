import numpy as np

import torch, gym, math

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

from IPython import display

%matplotlib inline
env = gym.make('CartPole-v0')
print("Action Space: {}".format(env.action_space))

print("State Space: {}".format(env.observation_space))
class Policy(nn.Module):

    def __init__(self, state_size, action_size):

        super(Policy, self).__init__()

        self.state_size = state_size

        self.action_size = action_size

        

        self.fc0 = nn.Linear(state_size, action_size)

        self.softmax = nn.Softmax(0)

    def forward(self, state):

        with torch.no_grad():

            x = self.softmax(self.fc0(state))

        return x

    def discounted_return(self, env, gamma=1.0, max_t=5000):

        with torch.no_grad():

            state = env.reset()

            dreturn = 0

            for i in range(max_t):

                probs = self.forward(torch.tensor(state).float())

                action = torch.multinomial(probs, 1)

                state, reward, done, _ = env.step(action.item())

                dreturn += gamma**i * reward

                if done:

                    break 

        return dreturn

    def set_weights(self, weight, bias):

        self.fc0.weight.data.copy_(weight)

        self.fc0.bias.data.copy_(bias)

        return self
epochs = 100

noise_scale = 10

noise_factor = 2

population = 10

policy = Policy(env.observation_space.shape[0], env.action_space.n)

best_weight = policy.fc0.weight.data

best_bias = policy.fc0.bias.data

best_return = -math.inf

scores = []

for i in range(epochs):

    with torch.no_grad():

        weights = torch.cat([(best_weight + noise_scale * best_weight.clone().normal_()).unsqueeze(0) for i in range(population)])

        biases = torch.cat([best_bias + noise_scale * best_bias.clone().normal_().unsqueeze(0) for i in range(population)])

        returns = torch.tensor([policy.set_weights(weight, bias.squeeze()).discounted_return(env) for weight, bias in zip(weights, biases)]).unsqueeze(1)

        average_weights = F.softmax(returns, 0)

        bias = (average_weights.T @ biases).squeeze()

        weight = (average_weights.unsqueeze(2) * weights).sum(0)

        policy.set_weights(weight, bias)

        mean_return = torch.tensor([policy.discounted_return(env) for i in range(10)]).mean()

        if mean_return > best_return:

            best_return = mean_return 

            best_weight = weight

            best_bias = bias

            noise_scale /= noise_factor

        else:

            noise_scale *= noise_factor

        scores.append(best_return)

        

plt.plot(scores)