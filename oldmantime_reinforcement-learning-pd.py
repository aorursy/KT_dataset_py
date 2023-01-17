import copy
import random
from IPython.display import clear_output
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column, row
output_notebook()
from IPython.display import Image
import os
Image("../input/rl-images/rl_agent.png", width=600, height=800)
state = [0 for i in range(9)]
state
state[0] = 1   # X plays action 0
state[3] = -1  # O plays action 3
state
def render(s):
    board = ['.' for i in range(9)]
    for i in range(9):
        if s[i] == 1: board[i] = 'X'
        if s[i] == -1: board[i] = 'O'
        if s[i] == 0: board[i] = ' '
    print(board[0], "|", board[1], "|", board[2])
    print("----------")
    print(board[3], "|", board[4], "|", board[5])
    print("----------")
    print(board[6], "|", board[7], "|", board[8])
    
render(state)
class ttt:
        
    def __init__(self):
        self.state = self.reset()
        
    def reset(self):
        return [0 for i in range(9)]

    def game_over(self, s):
        done = False
        reward = 0
        if (s[0] + s[1] + s[2]  == 3 or s[3] + s[4] + s[5]  == 3 or s[6] + s[7] + s[8]  == 3 or
            s[0] + s[3] + s[6]  == 3 or s[1] + s[4] + s[7]  == 3 or s[2] + s[5] + s[8]  == 3 or
            s[0] + s[4] + s[8]  == 3 or s[2] + s[4] + s[6]  == 3):
            done = True
            reward = 1
        if (s[0] + s[1] + s[2]  == -3 or s[3] + s[4] + s[5]  == -3 or s[6] + s[7] + s[8]  == -3 or
            s[0] + s[3] + s[6]  == -3 or s[1] + s[4] + s[7]  == -3 or s[2] + s[5] + s[8]  == -3 or
            s[0] + s[4] + s[8]  == -3 or s[2] + s[4] + s[6]  == -3):
            done = True
            reward = -1
        if sum(1 for i in s if i != 0)==9 and not done:
            done = True
        return done, reward

    def step(self, state, action, player):
        next_state = state.copy()
        if player == 0: next_state[action] = 1
        else: next_state[action] = -1
        done, reward = self.game_over(next_state)
        return next_state, done, reward
random.seed(1)
env = ttt()           # initialize the environment
state = env.reset()   # reset the game board

print("Start game")
render(state)         # display the game board
print(" ")

done = False
while not done:       # loop to play one game
    action = random.choice([i for i in range(len(state)) if state[i] == 0]) # Player X's move
    next_state, done, reward = env.step(state, action, 0)
    if not done:                                                            # Player O's move
        omove = random.choice([i for i in range(len(next_state)) if next_state[i] == 0])
        next_state, done, reward = env.step(next_state, omove, 1)
    state = next_state.copy()
    print("Action:", action, "Reward:", reward)
    render(state)
    print(" ")
def play_v_random (games, Qvalues, render_game=False):
    results = [0 for i in range(games)]
    for i in range(games):
        state = env.reset()
        done = False
        while not done:
            xq = [Qvalues.get((tuple(state), i)) for i in range(9) if Qvalues.get((tuple(state), i)) is not None]
            if len(xq) == 0: 
                action = random.choice([i for i in range(len(state)) if state[i] == 0])
            else:
                idx = [i for i in range(9) if Qvalues.get((tuple(state), i)) is not None]
                action = idx[xq.index(max(xq))]
            next_state, done, reward = env.step(state, action, 0)
            if not done:
                omove = random.choice([i for i in range(len(next_state)) if next_state[i] == 0])
                next_state, done, reward = env.step(next_state, omove, 1)
            state = next_state.copy()
            if render_game:
                print("Action:", action, "Reward:", reward)
                render(state)
                print(" ")
        results[i] = reward
    return results
random.seed(0)
alpha = 0.05            # learning rate
gamma = 0.95            # discount factor
Qvalues = {}            # Q-value dictionary
iterations = 500000

results = play_v_random(1000, Qvalues)
print("X Won: {:.1%}\tO Won: {:.1%}\tTies: {:.1%}".format(sum(1 for i in results if i == 1)/1000, 
                                                         sum(1 for i in results if i == -1)/1000, 
                                                         sum(1 for i in results if i == 0)/1000))

for iteration in range(iterations):    # loop to play a bunch of games
    state = env.reset()
    next_state = state.copy()
    done = False
    epsilon = max(1 - iteration/(iterations*0.8), 0.01)
    while not done:                    # loop to play one game
        if random.random() < epsilon:  # epsilon greedy policy for player X
            action = random.choice([i for i in range(len(state)) if state[i] == 0])
        else:
            xq = [Qvalues.get((tuple(state), i)) for i in range(9) if Qvalues.get((tuple(state), i)) is not None]
            if len(xq) == 0: action = random.choice([i for i in range(len(state)) if state[i] == 0])
            else:
                idx = [i for i in range(9) if Qvalues.get((tuple(state), i)) is not None]
                action = idx[xq.index(max(xq))]
        next_state, done, reward = env.step(state, action, 0)
        if not done:                  # random policy for player O
            omove = random.choice([i for i in range(len(next_state)) if next_state[i] == 0])
            next_state, done, reward = env.step(next_state, omove, 1)
        if not done:
            key = (tuple(state), action)
            if key not in Qvalues: Qvalues[key] = reward
            next_idx = [i for i in range(9) if Qvalues.get((tuple(next_state), i)) is not None]
            if len(next_idx) > 0: next_value = max([Qvalues.get((tuple(next_state), i)) for i in next_idx])
            else: next_value = 0
        else: next_value = reward
        # update the Q-value for the state-action pair
        Qvalues[key] *= 1 - alpha
        Qvalues[key] += alpha * (reward + gamma * next_value) 
        state = next_state.copy()
        
    if iteration % 50000 == 0:
        results = play_v_random(1000, Qvalues)
        print("X Won: {:.1%}\tO Won: {:.1%}\tTies: {:.1%}".format(sum(1 for i in results if i == 1)/1000, 
                                                                 sum(1 for i in results if i == -1)/1000, 
                                                                 sum(1 for i in results if i == 0)/1000))
random.seed(1)
play_v_random(1, Qvalues, render_game=True)
x = [i for i in range(len(Qvalues))]
q = q = list(Qvalues.values())
q.sort()
p = figure(title="Q Values", plot_height=300)
p.circle(x, q)
show(p)
!pip3 install ann_visualizer
import keras
from keras.models import Sequential
from keras.layers import Dense

network = Sequential()

# Hidden Layer #1
network.add(Dense(units=10, activation='relu', input_dim=6))

# Hidden Layer #2
network.add(Dense(units=10, activation='relu'))

# Output Layer
network.add(Dense(units=6, activation='softmax')) # softmax for categorical output (converts to probabilities)

from ann_visualizer.visualize import ann_viz

ann_viz(network, title="Example Fully Connected Network")
Image("../input/network/network.png", width = 1000, height = 400)
x = [i/10 for i in range(-10,10)]
y = [val if val>=0 else 0 for val in x]

p = figure(title="ReLu Activation", plot_height=300)
p.line(x, y)
show(p)
import numpy as np

weights = [i for i in range(-6, 7)]

prob = np.exp(weights)/sum(np.exp(weights))
print(prob)
print(sum(prob)) # sum to one?
p = figure(title="Softmax Activation", plot_height=300)
p.line(weights, prob)
show(p)
network.trainable_weights
import tensorflow as tf

class DQNagent:
    
    def __init__(self, state_size, action_size, iterations):
        self.gamma = 0.95                                    # discount factor
        self.state_size = state_size                         # 9 for tic-tac-toe
        self.action_size = action_size                       # 9 for tic-tac-toe
        self.iterations = iterations
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.SGD(lr=0.02)    # lr = learning rate (= alpha)
        self.loss_fn = tf.keras.losses.mean_squared_error

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.state_size**2, activation="relu", input_shape=[self.state_size]),
            tf.keras.layers.Dense(self.state_size**2, activation="relu"),
            tf.keras.layers.Dense(self.action_size)
        ])
        return model
    
    def train_model(self, state_history, action_history, next_state_history, rewards, dones):
        next_Q_values = self.model.predict(np.array(next_state_history))                         # 1. the forward pass
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - 1*np.array(dones)) * self.gamma * max_next_Q_values
        target_Q_values = tf.reshape(target_Q_values, [len(rewards), 1])
        mask = tf.one_hot(action_history, 9)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(np.array(state_history))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))                      # 2. measure the error
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))              # 3. the reverse pass
        
    def play_ttt(self):
        for iteration in range(self.iterations):    # outer loop to play the game a bunch of times
            state = env.reset()
            next_state = state.copy()
            done = False
            dones = []
            state_history = []
            state_history.append(state)
            action_history = []
            rewards = []
            epsilon = max(1 - iteration/(self.iterations*0.8), 0.01)
            while not done:                          # inner loop to play one game
                if random.random() < epsilon:        # epsilon-greedy policy
                    action = random.choice([i for i in range(len(state)) if state[i] == 0])
                else:
                    action = np.argmax(self.model.predict(np.array(state)[np.newaxis])[0])
                action_history.append(action)
                next_state, done, reward = env.step(state, action, 0)
                if done: 
                    state_history.append(next_state)
                    dones.append(done)
                    rewards.append(reward)
                if not done:
                    omove = random.choice([i for i in range(len(next_state)) if next_state[i] == 0])
                    next_state, done, reward = env.step(next_state, omove, 1)
                    state = next_state.copy()
                    state_history.append(next_state)
                    dones.append(done)
                    rewards.append(reward)
            next_state_history = state_history[1:len(state_history)]
            state_history = state_history[0:len(action_history)]
            self.train_model(state_history, action_history, next_state_history, rewards, dones)
        return self.model
def dqn_v_random (model, games, render_game=False):
    results = [0 for i in range(games)]
    for i in range(games):
        board = env.reset()
        done = False
        while not done:
            xmoves = model.predict(np.array(board)[np.newaxis])[0]
            xmoves[np.where(np.array(board)!=0)[0]] = -1
            xmove = np.argmax(xmoves)
            board[xmove] = 1
            done, reward = env.game_over(board)
            if not done:
                omove = random.choice([i for i in range(len(board)) if board[i] == 0])
                board[omove] = -1
                done, reward = env.game_over(board)
        results[i] = reward
        if render_game:
            print("Action:", action, "Reward:", reward)
            render(state)
            print(" ")
    return results
tf.random.set_seed(1234)
random.seed(1234)

m1 = DQNagent(9,9,1).play_ttt()

results = dqn_v_random(m1, 1000)

print("X Won: {:.1%}\tO Won: {:.1%}\tTies: {:.1%}".format(sum(1 for i in results if i == 1)/1000, 
                                                                 sum(1 for i in results if i == -1)/1000, 
                                                                 sum(1 for i in results if i == 0)/1000))
tf.keras.backend.clear_session()
tf.random.set_seed(1234)
random.seed(1234)

m100 = DQNagent(9,9,100).play_ttt()

results = dqn_v_random(m100, 1000)

print("X Won: {:.1%}\tO Won: {:.1%}\tTies: {:.1%}".format(sum(1 for i in results if i == 1)/1000, 
                                                                 sum(1 for i in results if i == -1)/1000, 
                                                                 sum(1 for i in results if i == 0)/1000))
tf.keras.backend.clear_session()
tf.random.set_seed(1234)
random.seed(1234)

m1000 = DQNagent(9,9,1000).play_ttt()

results = dqn_v_random(m1000, 1000)

print("X Won: {:.1%}\tO Won: {:.1%}\tTies: {:.1%}".format(sum(1 for i in results if i == 1)/1000, 
                                                                 sum(1 for i in results if i == -1)/1000, 
                                                                 sum(1 for i in results if i == 0)/1000))