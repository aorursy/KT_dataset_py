# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import time

import copy

import numpy as np

import pandas as pd

import chainer

import chainer.functions as F

import chainer.links as L

from plotly import tools

from plotly.graph_objs import *

from plotly.offline import init_notebook_mode, iplot, iplot_mpl

init_notebook_mode()
data = pd.read_csv("../input/bitmex-trades/bitmex.csv")

data['Date'] = pd.to_datetime(data['Date'])

data = data.set_index('Date')

print(data.index.min(), data.index.max())

data.head()
date_split = '2018-01-01'

train = data[:date_split]

test = data[date_split:]

test = test[:1500]

len(train), len(test)
def plot_train_test(train, test, date_split):

    

    data = [

        Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'], name='train'),

        Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'], name='test')

    ]

    layout = {

         'shapes': [

             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}

         ],

        'annotations': [

            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},

            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}

        ]

    }

    figure = Figure(data=data, layout=layout)

    iplot(figure)
plot_train_test(train, test, date_split)
class Environment1:

    

    def __init__(self, data, history_t=90):

        self.data = data

        self.history_t = history_t

        self.reset()

        

    def reset(self):

        self.t = 0

        self.done = False

        self.profits = 0

        self.positions = []

        self.position_value = 0

        self.history = [0 for _ in range(self.history_t)]

        return [self.position_value] + self.history # obs

    

    def step(self, act):

        reward = 0

        

        # act = 0: stay, 1: buy, 2: sell

        if act == 1:

            self.positions.append(self.data.iloc[self.t, :]['Close'])

        elif act == 2: # sell

            if len(self.positions) == 0:

                reward = -1

            else:

                profits = 0

                for p in self.positions:

                    profits += (self.data.iloc[self.t, :]['Close'] - p)

                reward += profits

                self.profits += profits

                self.positions = []

        

        # set next time

        self.t += 1

        self.position_value = 0

        for p in self.positions:

            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)

        self.history.pop(0)

        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])

        

        # clipping reward

        if reward > 0:

            reward = 1

        elif reward < 0:

            reward = -1

        

        return [self.position_value] + self.history, reward, self.done # obs, reward, done
env = Environment1(train)

print(env.reset())

for _ in range(3):

    pact = np.random.randint(3)

    print(env.step(pact))


def train_dqn(env):



    class Q_Network(chainer.Chain):



        def __init__(self, input_size, hidden_size, output_size):

            super(Q_Network, self).__init__(

                fc1 = L.Linear(input_size, hidden_size),

                fc2 = L.Linear(hidden_size, hidden_size),

                fc3 = L.Linear(hidden_size, output_size)

            )



        def __call__(self, x):

            h = F.relu(self.fc1(x))

            h = F.relu(self.fc2(h))

            y = self.fc3(h)

            return y



        def reset(self):

            self.zerograds()



    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)

    Q_ast = copy.deepcopy(Q)

    optimizer = chainer.optimizers.Adam()

    optimizer.setup(Q)



    epoch_num = 50

    step_max = len(env.data)-1

    memory_size = 200

    batch_size = 20

    epsilon = 1.0

    epsilon_decrease = 1e-3

    epsilon_min = 0.1

    start_reduce_epsilon = 200

    train_freq = 10

    update_q_freq = 20

    gamma = 0.97

    show_log_freq = 5



    memory = []

    total_step = 0

    total_rewards = []

    total_losses = []



    start = time.time()

    for epoch in range(epoch_num):



        pobs = env.reset()

        step = 0

        done = False

        total_reward = 0

        total_loss = 0



        while not done and step < step_max:



            # select act

            pact = np.random.randint(3)

            if np.random.rand() > epsilon:

                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))

                pact = np.argmax(pact.data)



            # act

            obs, reward, done = env.step(pact)



            # add memory

            memory.append((pobs, pact, reward, obs, done))

            if len(memory) > memory_size:

                memory.pop(0)



            # train or update q

            if len(memory) == memory_size:

                if total_step % train_freq == 0:

                    shuffled_memory = np.random.permutation(memory)

                    memory_idx = range(len(shuffled_memory))

                    for i in memory_idx[::batch_size]:

                        batch = np.array(shuffled_memory[i:i+batch_size])

                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)

                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)

                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)

                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)

                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)



                        q = Q(b_pobs)

                        maxq = np.max(Q_ast(b_obs).data, axis=1)

                        target = copy.deepcopy(q.data)

                        for j in range(batch_size):

                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])

                        Q.reset()

                        loss = F.mean_squared_error(q, target)

                        total_loss += loss.data

                        loss.backward()

                        optimizer.update()



                if total_step % update_q_freq == 0:

                    Q_ast = copy.deepcopy(Q)



            # epsilon

            if epsilon > epsilon_min and total_step > start_reduce_epsilon:

                epsilon -= epsilon_decrease



            # next step

            total_reward += reward

            pobs = obs

            step += 1

            total_step += 1



        total_rewards.append(total_reward)

        total_losses.append(total_loss)



        if (epoch+1) % show_log_freq == 0:

            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq

            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq

            elapsed_time = time.time()-start

            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))

            start = time.time()

            

    return Q, total_losses, total_rewards

    
Q, total_losses, total_rewards = train_dqn(Environment1(train))
def plot_loss_reward(total_losses, total_rewards):



    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=('loss', 'reward'), print_grid=False)

    figure.append_trace(Scatter(y=total_losses, mode='lines', line=dict(color='skyblue')), 1, 1)

    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1, 2)

    figure['layout']['xaxis1'].update(title='epoch')

    figure['layout']['xaxis2'].update(title='epoch')

    figure['layout'].update(height=400, width=900, showlegend=False)

    iplot(figure)

    
plot_loss_reward(total_losses, total_rewards)
Q, total_losses, total_rewards = train_dqn(Environment1(test))
plot_loss_reward(total_losses, total_rewards)