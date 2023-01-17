import plotly.graph_objects as go

import matplotlib.pyplot as plt

import numpy as np

import os
prices = np.loadtxt('../input/binance-bitcoin-futures-price-10s-intervals/prices_btc_Jan_11_2020_to_May_22_2020.txt', dtype=float)
len(prices)
fig = go.Figure(data=go.Scatter(y=prices[-10000:]))

fig.show()



def saludo():

    print("Hola amigas")

    

saludo()
def buy(btc_price, btc, money):

    if(money != 0):

        btc = (1 / btc_price ) * money

        money = 0

    return btc, money





def sell(btc_price, btc, money):

    if(btc != 0):

        money = btc_price * btc

        btc = 0

    return btc, money





def wait(btc_price, btc, money):

    # do nothing

    return btc, money
np.random.seed(1)



# set of actions that the user could do

actions = { 'buy' : buy, 'sell': sell, 'wait' : wait}



actions_to_nr = { 'buy' : 0, 'sell' : 1, 'wait' : 2 }

nr_to_actions = { k:v for (k,v) in enumerate(actions_to_nr) }



nr_actions = len(actions_to_nr.keys())

nr_states = len(prices)



# q-table = reference table for our agent to select the best action based on the q-value

q_table = np.random.rand(nr_states, nr_actions)
def get_reward(before_btc, btc, before_money, money):

    reward = 0

    if(btc != 0):

        if(before_btc < btc):

            reward = 1

    if(money != 0):

        if(before_money < money):

            reward = 1

            

    return reward
def choose_action(state):

    if np.random.uniform(0, 1) < eps:

        return np.random.randint(0, 2)

    else:

        return np.argmax(q_table[state])
def take_action(state, action):

    return actions[nr_to_actions[action]](prices[state], btc, money)
def act(state, action, theta):

    btc, money = theta

    

    done = False

    new_state = state + 1

    

    before_btc, before_money = btc, money

    btc, money = take_action(state, action)

    theta = btc, money

    

    reward = get_reward(before_btc, btc, before_money, money)

    

    if(new_state == nr_states):

        done = True

    

    return new_state, reward, theta, done
reward = 0

btc = 0

money = 100



theta = btc, money
# exploratory

eps = 0.3



n_episodes = 20

min_alpha = 0.02



# learning rate for Q learning

alphas = np.linspace(1.0, min_alpha, n_episodes)



# discount factor, used to balance immediate and future reward

gamma = 1.0
rewards = {}



for e in range(n_episodes):

    

    total_reward = 0

    

    state = 0

    done = False

    alpha = alphas[e]

    

    while(done != True):



        action = choose_action(state)

        next_state, reward, theta, done = act(state, action, theta)

        

        total_reward += reward

        

        if(done):

            rewards[e] = total_reward

            print(f"Episode {e + 1}: total reward -> {total_reward}")

            break

        

        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma *  np.max(q_table[next_state]) - q_table[state][action])



        state = next_state
plt.ylabel('Total Reward')

plt.xlabel('Episode')

plt.plot([rewards[e] for e in rewards.keys()])
state = 0

acts = np.zeros(nr_states)

done = False



while(done != True):



        action = choose_action(state)

        next_state, reward, theta, done = act(state, action, theta)

        

        acts[state] = action

        

        total_reward += reward

        

        if(done):

            break

            

        state = next_state
buys_idx = np.where(acts == 0)

wait_idx = np.where(acts == 2)

sell_idx = np.where(acts == 1)
plt.figure(figsize=(15,15))

plt.plot(buys_idx[0], prices[buys_idx], 'bo', markersize=2)

plt.plot(sell_idx[0], prices[sell_idx], 'ro', markersize=2)

plt.plot(wait_idx[0], prices[wait_idx], 'yo', markersize=2)