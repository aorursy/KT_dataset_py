import numpy as np

import random

import matplotlib.pyplot as plt
class ContextBandit:

    def __init__(self, arms=10):

        self.arms = arms

        self.init_distribution(arms)

        self.update_state()

        

    def init_distribution(self, arms):

        self.bandit_matrix = np.random.rand(arms,arms)

        

    def reward(self, prob):

        reward = 0

        for i in range(self.arms):

            if random.random() < prob:

                reward += 1

        return reward

        

    def get_state(self):

        return self.state



    def update_state(self):

        self.state = np.random.randint(0,self.arms)

        

    def get_reward(self,arm):

        return self.reward(self.bandit_matrix[self.get_state()][arm])

        

    def choose_arm(self, arm):

        reward = self.get_reward(arm)

        self.update_state()

        return reward
env = ContextBandit(arms=10)

state = env.get_state()

reward = env.choose_arm(1)

print('State ',state)

print('Reward ',reward)
arms = 10

N, D_in, H, D_out = 1, arms, 100, arms
def relu(x):

    return x * (x > 0)



def drelu(x):

    return 1. * (x > 0)
def mse(y, y_hat):

    return np.mean((y - y_hat) ** 2)



def dmse(y, y_hat):

    return np.mean(y - y_hat)
def softmax(av, tau):

    return np.exp(av / tau) / np.sum( np.exp(av / tau) )
def one_hot(N, pos, val=1):

    one_hot_vec = np.zeros(N)

    one_hot_vec[pos] = val

    return one_hot_vec
def forward(x, alpha):

    wi, wo = alpha

    

    wil = x.reshape(1,10).dot(wi)

    wilr = relu(wil)

    

    wol = wilr.dot(wo)

    wolr = relu(wol)

    return wil, wilr, wol, wolr
def backward(x, pred, y, alpha, params):

    wil, wilr, wol, wolr = params

    wi, wo = alpha

    

    pred = pred.reshape(10)

    dloss = dmse(pred, y)

    

    # dloss * drelu(wol) * wilr

    dwo = np.dot(wilr.T, dloss * drelu(wol))

    # dloss * drelu(wol) * wo * drelu(wil) * x

    dwi = np.dot(x.reshape(10,1), np.dot((dloss * drelu(wol)), (wo * drelu(wil).T).T))

    

    return dwi, dwo
def SGE(alpha, grads, lr=1e-4):

    wi, wo = alpha

    dwi, dwo = grads

    

    wi += lr * dwi

    wo += lr * dwo

    

    return wi, wo
wi = np.random.uniform(0, 0.5, size=(D_in, H))

wo = np.random.uniform(0, 0.5, size=(H, D_out))



alpha = wi, wo
rewards = []

temp_rewards = []

env = ContextBandit(arms=10)

cur_state = one_hot(arms, env.get_state())
def get_choice(params):

    av_softmax = softmax(params[-1], tau=np.max(params[-1]))

    av_softmax /= av_softmax.sum()



    return np.random.choice(arms, p=av_softmax.reshape(10))
for epoch in range(5000):

    

    params = forward(cur_state, alpha)



    choice = get_choice(params)



    cur_reward = env.choose_arm(choice)



    one_hot_reward = params[-1].reshape(10).copy()

    one_hot_reward[choice] = cur_reward



    reward = one_hot_reward

    temp_rewards.append(reward)



    loss = mse(params[-1], reward)

    

    if(epoch % 1000 == 0):

        mean_rewards = np.mean(temp_rewards)

        print("Mean rewards :", mean_rewards)

        rewards.append(mean_rewards)

        temp_rewards = []



    dwi, dwo = backward(cur_state, params[-1], reward, alpha, params)

    dwi = np.clip(dwi, -1, 1)

    dwo = np.clip(dwo, -1, 1)

    grads = dwi, dwo

    

    alpha = SGE(alpha, grads)

    

    cur_state = one_hot(arms, env.get_state())
plt.title('Mean rewards over epochs')

plt.plot(rewards, 'bo')
cur_state = one_hot(arms, env.get_state())

params = forward(cur_state, alpha)

choice = get_choice(params)

cur_reward = env.choose_arm(choice)

print('State: %d, Choice: %d, Reward: %d' % (np.argmax(cur_state), choice, cur_reward))