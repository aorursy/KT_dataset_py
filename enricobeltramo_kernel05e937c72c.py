import numpy as np

import pandas as pd

import time

import copy

import chainer

import chainer.functions as F

import chainer.links as L

from plotly import tools

from plotly.graph_objs import *

from plotly.offline import init_notebook_mode, iplot, iplot_mpl

from kaggle.competitions import twosigmanews



from rl.agents.dqn import DQNAgent

from rl.policy import EpsGreedyQPolicy

from rl.memory import SequentialMemory



import gym



env = twosigmanews.make_env()

(marketdf, newsdf) = env.get_training_data()





print('preparing data...')

def prepare_data(marketdf, newsdf):

    # a bit of feature engineering

    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)

    marketdf['bartrend'] = marketdf['close'] / marketdf['open']

    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2

    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']

    

    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)

    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])

    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']

    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']



    # filter pre-2012 data, no particular reason

    marketdf = marketdf.loc[marketdf['time'] > 20120000]

    

    # get rid of extra junk from news data

    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',

                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',

                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']

    newsdf.drop(droplist, axis=1, inplace=True)

    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)

    

    # combine multiple news reports for same assets on same day

    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()

    

    # join news reports to market data, note many assets will have many days without news data

    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])



cdf = prepare_data(marketdf, newsdf)    

del marketdf, newsdf  # save the precious memory
cdf.head()
assetsCode = cdf['assetCode'].unique()

targetcols = ['returnsOpenNextMktres10']

traincols = [col for col in cdf.columns if col not in ['time', 

                                                        'universe', 

                                                        'noveltyCount12H',

                                                        'noveltyCount24H',

                                                        'noveltyCount3D',

                                                        'noveltyCount5D',

                                                        'noveltyCount7D',

                                                        'volumeCounts12H',

                                                        'volumeCounts24H',

                                                        'volumeCounts3D',

                                                        'volumeCounts5D',

                                                        'volumeCounts7D',

                                                        'position',

                                                        'coverage',

                                                        'bartrend',

                                                        'average',

                                                        'pricevolume',

                                                        'companyCount',

                                                        'relevance',] + targetcols]



dates = cdf['time'].unique()

train = range(len(dates))[:int(0.85*len(dates))]

val = range(len(dates))[int(0.85*len(dates)):]



# we be classifyin

cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)



# train data

filter_t = cdf['time'].isin(dates[train]) & cdf['assetCode'].isin([assetsCode[0]])

Xt = cdf[traincols].fillna(0).loc[filter_t]

Yt = cdf[targetcols].fillna(0).loc[filter_t]



Xt.head()



Xt = Xt.values

Yt = Yt.values





# validation data

filter_v = cdf['time'].isin(dates[val]) & cdf['assetCode'].isin([assetsCode[0]])

Xv = cdf[traincols].fillna(0).loc[filter_v]

Yv = cdf[targetcols].fillna(0).loc[filter_v]



Xv.head()



Xv = Xv.values

Yv = Yv.values



print(Xt.shape, Xv.shape)
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
# DQN



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
def plot_train_test_by_q(train_env, test_env, Q, algorithm_name):

    

    # train

    pobs = train_env.reset()

    train_acts = []

    train_rewards = []



    for _ in range(len(train_env.data)-1):

        

        pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))

        pact = np.argmax(pact.data)

        train_acts.append(pact)

            

        obs, reward, done = train_env.step(pact)

        train_rewards.append(reward)



        pobs = obs

        

    train_profits = train_env.profits

    

    # test

    pobs = test_env.reset()

    test_acts = []

    test_rewards = []



    for _ in range(len(test_env.data)-1):

    

        pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))

        pact = np.argmax(pact.data)

        test_acts.append(pact)

            

        obs, reward, done = test_env.step(pact)

        test_rewards.append(reward)



        pobs = obs

        

    test_profits = test_env.profits

    

    # plot

    train_copy = train_env.data.copy()

    test_copy = test_env.data.copy()

    train_copy['act'] = train_acts + [np.nan]

    train_copy['reward'] = train_rewards + [np.nan]

    test_copy['act'] = test_acts + [np.nan]

    test_copy['reward'] = test_rewards + [np.nan]

    train0 = train_copy[train_copy['act'] == 0]

    train1 = train_copy[train_copy['act'] == 1]

    train2 = train_copy[train_copy['act'] == 2]

    test0 = test_copy[test_copy['act'] == 0]

    test1 = test_copy[test_copy['act'] == 1]

    test2 = test_copy[test_copy['act'] == 2]

    act_color0, act_color1, act_color2 = 'gray', 'cyan', 'magenta'



    data = [

        Candlestick(x=train0.index, open=train0['Open'], high=train0['High'], low=train0['Low'], close=train0['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),

        Candlestick(x=train1.index, open=train1['Open'], high=train1['High'], low=train1['Low'], close=train1['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),

        Candlestick(x=train2.index, open=train2['Open'], high=train2['High'], low=train2['Low'], close=train2['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2))),

        Candlestick(x=test0.index, open=test0['Open'], high=test0['High'], low=test0['Low'], close=test0['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),

        Candlestick(x=test1.index, open=test1['Open'], high=test1['High'], low=test1['Low'], close=test1['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),

        Candlestick(x=test2.index, open=test2['Open'], high=test2['High'], low=test2['Low'], close=test2['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2)))

    ]

    title = '{}: train s-reward {}, profits {}, test s-reward {}, profits {}'.format(

        algorithm_name,

        int(sum(train_rewards)),

        int(train_profits),

        int(sum(test_rewards)),

        int(test_profits)

    )

    layout = {

        'title': title,

        'showlegend': False,

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
plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'DQN')
# Double DQN



def train_ddqn(env):



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

    batch_size = 50

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

                        """ <<< DQN -> Double DQN

                        maxq = np.max(Q_ast(b_obs).data, axis=1)

                        === """

                        indices = np.argmax(q.data, axis=1)

                        maxqs = Q_ast(b_obs).data

                        """ >>> """

                        target = copy.deepcopy(q.data)

                        for j in range(batch_size):

                            """ <<< DQN -> Double DQN

                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])

                            === """

                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])

                            """ >>> """

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
Q, total_losses, total_rewards = train_ddqn(Environment1(train))
plot_loss_reward(total_losses, total_rewards)
plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Double DQN')
# Dueling Double DQN



def train_dddqn(env):



    """ <<< Double DQN -> Dueling Double DQN

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

    === """

    class Q_Network(chainer.Chain):



        def __init__(self, input_size, hidden_size, output_size):

            super(Q_Network, self).__init__(

                fc1 = L.Linear(input_size, hidden_size),

                fc2 = L.Linear(hidden_size, hidden_size),

                fc3 = L.Linear(hidden_size, hidden_size//2),

                fc4 = L.Linear(hidden_size, hidden_size//2),

                state_value = L.Linear(hidden_size//2, 1),

                advantage_value = L.Linear(hidden_size//2, output_size)

            )

            self.input_size = input_size

            self.hidden_size = hidden_size

            self.output_size = output_size



        def __call__(self, x):

            h = F.relu(self.fc1(x))

            h = F.relu(self.fc2(h))

            hs = F.relu(self.fc3(h))

            ha = F.relu(self.fc4(h))

            state_value = self.state_value(hs)

            advantage_value = self.advantage_value(ha)

            advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)

            q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))

            return q_value



        def reset(self):

            self.zerograds()

    """ >>> """



    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)

    Q_ast = copy.deepcopy(Q)

    optimizer = chainer.optimizers.Adam()

    optimizer.setup(Q)



    epoch_num = 50

    step_max = len(env.data)-1

    memory_size = 200

    batch_size = 50

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

                        """ <<< DQN -> Double DQN

                        maxq = np.max(Q_ast(b_obs).data, axis=1)

                        === """

                        indices = np.argmax(q.data, axis=1)

                        maxqs = Q_ast(b_obs).data

                        """ >>> """

                        target = copy.deepcopy(q.data)

                        for j in range(batch_size):

                            """ <<< DQN -> Double DQN

                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])

                            === """

                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])

                            """ >>> """

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
Q, total_losses, total_rewards = train_dddqn(Environment1(train))
plot_loss_reward(total_losses, total_rewards)
plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Dueling Double DQN')





lgtrain, lgval = lgb.Dataset(Xt, Yt[:,0]), lgb.Dataset(Xv, Yv[:,0])

lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=200)





############################################################

print("generating predictions...")

preddays = env.get_prediction_days()

for marketdf, newsdf, predtemplatedf in preddays:

    cdf = prepare_data(marketdf, newsdf)

    Xp = cdf[traincols].fillna(0).values

    preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1

    predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':preds})

    predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values

    env.predict(predtemplatedf)



env.write_submission_file()