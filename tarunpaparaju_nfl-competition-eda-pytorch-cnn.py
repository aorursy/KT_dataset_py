import os

import gc

import time

import math

import tqdm

import numpy as np

import pandas as pd

from keras.utils import to_categorical

from tqdm import tqdm_notebook as tqdm

from kaggle.competitions import nflrush



env = nflrush.make_env()

iter_test = env.iter_test()



from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler



!pip install -qqq hiddenlayer

import hiddenlayer as hl



import IPython

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff





import torch

import torch.nn as nn

from torch.optim import Adam

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



import warnings

warnings.filterwarnings('ignore')
EPOCHS = 8

BATCH_SIZE = 128

DATA_PATH = '../input/nfl-big-data-bowl-2020/'
train_df = pd.read_csv(DATA_PATH + 'train.csv')

train_df.head()
fig = ff.create_distplot(hist_data=[train_df.sample(frac=0.025)["X"]], group_labels="X", colors=['rgb(26, 153, 0)'])

fig.update_layout(title="X coordinate", yaxis=dict(title="Probability Density"), xaxis=dict(title="X coordinate"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["X"], y=data["Yards"], kind='kde', color='forestgreen', height=7)

plot.set_axis_labels('X coordinate', 'Yards', fontsize=16)

plt.show(plot)
fig = ff.create_distplot(hist_data=[train_df.sample(frac=0.025)["Y"]], group_labels="Y", colors=['rgb(179, 0, 30)'])

fig.update_layout(title="Y coordinate", yaxis=dict(title="Probability Density"), xaxis=dict(title="Y coordinate"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["Y"], y=data["Yards"], kind='kde', color=(179/255, 0, 30/255), height=7)

plot.set_axis_labels('Y coordinate', 'Yards', fontsize=16)

plt.show(plot)
data = train_df.sample(frac=0.025)

plot = sns.jointplot(x=data["X"], y=data["Y"], kind='kde', color='mediumvioletred', height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
data = train_df
empty_data = data.query('OffenseFormation == "EMPTY"')

iform_data = data.query('OffenseFormation == "I_FORM"')

jumbo_data = data.query('OffenseFormation == "JUMBO"')

pistol_data = data.query('OffenseFormation == "PISTOL"')

shotgun_data = data.query('OffenseFormation == "SHOTGUN"')

singleback_data = data.query('OffenseFormation == "SINGLEBACK"')

wildcat_data = data.query('OffenseFormation == "WILDCAT"')
colors = ['red', 'orangered', 'orange', (179/255, 149/255, 0), 'forestgreen', 'blue', 'blueviolet', 'darkviolet']
plot = sns.jointplot(x=empty_data["X"], y=empty_data["Y"], kind='kde', color=colors[0], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=iform_data["X"], y=iform_data["Y"], kind='kde', color=colors[1], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=jumbo_data["X"], y=jumbo_data["Y"], kind='kde', color=colors[2], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=pistol_data["X"], y=pistol_data["Y"], kind='kde', color=colors[3], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=shotgun_data["X"], y=shotgun_data["Y"], kind='kde', color=colors[4], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=singleback_data["X"], y=singleback_data["Y"], kind='kde', color=colors[5], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
plot = sns.jointplot(x=wildcat_data["X"], y=wildcat_data["Y"], kind='kde', color=colors[6], height=7)

plot.set_axis_labels('X coordinate', 'Y coordinate', fontsize=16)

plt.show(plot)
fig = ff.create_distplot(hist_data=[train_df.sample(frac=0.025).query('Dir == Dir')["Dir"]], group_labels=["Dir"], colors=['rgb(255, 102, 25)'])

fig.update_layout(title="Dir", yaxis=dict(title="Probability Density"), xaxis=dict(title="Dir"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["Dir"], y=data["Yards"], kind='kde', color=(255/255, 102/255, 25/255), height=7)

plot.set_axis_labels('Dir', 'Yards', fontsize=16)

plt.show(plot)
fig = ff.create_distplot(hist_data=[train_df.sample(frac=0.025)["A"]], group_labels="A", colors=['rgb(0, 0, 230)'])

fig.update_layout(title="A", yaxis=dict(title="Probability Density"), xaxis=dict(title="A"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["A"], y=data["Yards"], kind='kde', color=(0, 0, 230/255), height=7)

plot.set_axis_labels('A', 'Yards', fontsize=16)

plt.show(plot)
fig = ff.create_distplot(hist_data=[train_df.sample(frac=0.025)["S"]], group_labels="S", colors=['rgb(230, 0, 191)'])

fig.update_layout(title="S", yaxis=dict(title="Probability Density"), xaxis=dict(title="S"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["S"], y=data["Yards"], kind='kde', color=(230/255, 0, 191/255), height=7)

plot.set_axis_labels('S', 'Yards', fontsize=16)

plt.show(plot)
data = train_df.sample(frac=0.025)["Humidity"]

fig = ff.create_distplot(hist_data=[data.fillna(data.mean())], group_labels=["Humidity"], colors=['rgb(0, 102, 102)'])

fig.update_layout(title="Humidity", yaxis=dict(title="Probability Density"), xaxis=dict(title="Humidity"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["Humidity"], y=data["Yards"], kind='kde', color=(0/255, 77/255, 77/255), height=7)

plot.set_axis_labels('Humidity', 'Yards', fontsize=16)

plt.show(plot)
data = train_df.sample(frac=0.025)["Temperature"]

fig = ff.create_distplot(hist_data=[data.fillna(data.mean())], group_labels=["Temperature"], colors=['rgb(51, 34, 0)'])

fig.update_layout(title="Temperature", yaxis=dict(title="Probability Density"), xaxis=dict(title="Temperature"))
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

plot = sns.jointplot(x=data["Temperature"], y=data["Yards"], kind='kde', color=(51/255, 34/255, 0), height=7)

plot.set_axis_labels('Temperature', 'Yards', fontsize=16)

plt.show(plot)
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

away_data = data.query('Team == "away"')["Yards"]

home_data = data.query('Team == "home"')["Yards"]



fig = ff.create_distplot(hist_data=[away_data, home_data],

                         group_labels=["Away", "Home"],

                         show_hist=False)



fig.update_layout(title="Team vs. Yards", yaxis=dict(title="Probability Density"), xaxis=dict(title="Yards"))

fig.show()
fig = go.Figure()

data = [away_data, home_data]

tags = ["Away", "Home"]



for index, category in enumerate(data):

    fig.add_trace(go.Box(y=category, name=tags[index]))



fig.update_layout(title="Team vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="Team"))

fig.show()
fig = go.Figure()

data = [away_data, home_data]

tags = ["Away", "Home"]



for index, category in enumerate(data):

    fig.add_trace(go.Violin(y=category, name=tags[index]))



fig.update_layout(title="Team vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="Team"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

north_data = data.query('WindDirection == "N"')["Yards"]

east_data = data.query('WindDirection == "E"')["Yards"]

west_data = data.query('WindDirection == "W"')["Yards"]

south_data = data.query('WindDirection == "S"')["Yards"]



fig = ff.create_distplot(hist_data=[north_data, east_data, west_data, south_data],

                         group_labels=["North", "East", "West", "South"],

                         show_hist=False)



fig.update_layout(title="WindDirection vs. Yards", yaxis=dict(title="Probability Density"), xaxis=dict(title="Yards"))

fig.show()
fig = go.Figure()

data = [north_data, east_data, west_data, south_data]

tags = ["North", "East", "West", "South"]



for index, category in enumerate(data):

    fig.add_trace(go.Box(y=category, name=tags[index]))

    

fig.update_layout(title="WindDirection vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="WindDirection"))

fig.show()
fig = go.Figure()

data = [north_data, east_data, west_data, south_data]

tags = ["North", "East", "West", "South"]



for index, category in enumerate(data):

    fig.add_trace(go.Violin(y=category, name=tags[index]))



fig.update_layout(title="WindDirection vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="WindDirection"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

empty_data = data.query('OffenseFormation == "EMPTY"')["Yards"]

iform_data = data.query('OffenseFormation == "I_FORM"')["Yards"]

jumbo_data = data.query('OffenseFormation == "JUMBO"')["Yards"]

pistol_data = data.query('OffenseFormation == "PISTOL"')["Yards"]

shotgun_data = data.query('OffenseFormation == "SHOTGUN"')["Yards"]

singleback_data = data.query('OffenseFormation == "SINGLEBACK"')["Yards"]

wildcat_data = data.query('OffenseFormation == "WILDCAT"')["Yards"]



fig = ff.create_distplot(hist_data=[empty_data, iform_data, jumbo_data,

                                    pistol_data, shotgun_data, singleback_data, wildcat_data],

                         group_labels=["Empty", "I-Form", "Jumbo", "Pistol",

                                       "Shotgun", "Singleback", "Wildcat"],

                         show_hist=False)



fig.update_layout(title="OffenseFormation vs. Yards", yaxis=dict(title="Probability Density"), xaxis=dict(title="Yards"))

fig.show()
fig = go.Figure()

data = [empty_data, iform_data, jumbo_data, pistol_data, shotgun_data, singleback_data, wildcat_data]

tags = ["Empty", "I-Form", "Jumbo", "Pistol", "Shotgun", "Singleback", "Wildcat"]



for index, category in enumerate(data):

    fig.add_trace(go.Box(y=category, name=tags[index]))



fig.update_layout(title="OffenseFormation vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="OffenseFormation"))

fig.show()
fig = go.Figure()

data = [empty_data, iform_data, jumbo_data, pistol_data, shotgun_data, singleback_data, wildcat_data]

tags = ["Empty", "I-Form", "Jumbo", "Pistol", "Shotgun", "Singleback", "Wildcat"]



for index, category in enumerate(data):

    fig.add_trace(go.Violin(y=category, name=tags[index]))



fig.update_layout(title="OffenseFormation vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="OffenseFormation"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

hist_data = [data.loc[data["HomeTeamAbbr"] == home_team_abbr]["Yards"] for home_team_abbr in set(data['HomeTeamAbbr'])]



fig = ff.create_distplot(hist_data=hist_data, group_labels=list(set(data['HomeTeamAbbr'])), show_hist=False)

fig.update_layout(title="HomeTeamAbbr vs. Yards", yaxis=dict(title="Probability Density"), xaxis=dict(title="Yards"))

fig.show()
hist_data = [ele for ele in reversed(hist_data)] 

tags = [ele for ele in reversed(list(set(data['HomeTeamAbbr'])))]
fig = go.Figure()



for index, category in enumerate(hist_data):

    fig.add_trace(go.Box(y=category, name=tags[index]))



fig.update_layout(title="HomeTeamAbbr vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="HomeTeamAbbr"))

fig.show()
fig = go.Figure()



for index, category in enumerate(hist_data):

    fig.add_trace(go.Violin(y=category, name=tags[index]))



fig.update_layout(title="HomeTeamAbbr vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="HomeTeamAbbr"))

fig.show()
data = train_df.sample(frac=0.025)

quantile = data["Yards"].quantile(0.95)

data = data.loc[data["Yards"] < quantile]

hist_data = [data.loc[data["VisitorTeamAbbr"] == visitor_team_abbr]["Yards"] for visitor_team_abbr in set(data['VisitorTeamAbbr'])]



fig = ff.create_distplot(hist_data=hist_data, group_labels=list(set(data['VisitorTeamAbbr'])), show_hist=False)

fig.update_layout(title="VisitorTeamAbbr vs. Yards", yaxis=dict(title="Probability Density"), xaxis=dict(title="Yards"))

fig.show()
hist_data = [ele for ele in reversed(hist_data)]

tags = [ele for ele in reversed(list(set(data['VisitorTeamAbbr'])))]
fig = go.Figure()



for index, category in enumerate(hist_data):

    fig.add_trace(go.Box(y=category, name=tags[index]))



fig.update_layout(title="VisitorTeamAbbr vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="VisitorTeamAbbr"))

fig.show()
fig = go.Figure()



for index, category in enumerate(hist_data):

    fig.add_trace(go.Violin(y=category, name=tags[index]))



fig.update_layout(title="VisitorTeamAbbr vs. Yards", yaxis=dict(title="Yards"), xaxis=dict(title="VisitorTeamAbbr"))

fig.show()
cat_cols = ['Team', 'FieldPosition', 'OffenseFormation']

value_dicts = []



for feature in cat_cols:

    values = set(train_df[feature])

    value_dicts.append(dict(zip(values, np.arange(len(values)))))
def indices(data, feat_index):

    value_dict = value_dicts[feat_index]

    return data[cat_cols[feat_index]].apply(lambda x: value_dict[x])



def one_hot(indices, feat_index):

    return to_categorical(indices, num_classes=len(value_dicts[feat_index]))
def get_categorical_features(sample, data):

    index_values = [indices(sample, index) for index in range(len(value_dicts))]

    features = tuple([one_hot(value, index) for index, value in enumerate(index_values)])

    features = np.concatenate(features, axis=1)

    return features
num_cols = ['X', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine',

            'Quarter', 'Down', 'Distance', 'HomeScoreBeforePlay',

            'VisitorScoreBeforePlay', 'DefendersInTheBox', 'PlayerWeight',

            'Week', 'Temperature', 'Humidity']



def get_numerical_features(sample):

    return sample[num_cols].values
class NFLCompetitionDataset(Dataset):

    

    def __init__(self, data, stage):

        self.dataframe = data

        self.stage = stage

        self.play_ids = list(set(data['PlayId']))

            

    def __len__(self):

        return len(self.play_ids)

        

    def __getitem__(self, index):

        data_locations = self.dataframe['PlayId'] == self.play_ids[index]

        data_sample = self.dataframe.loc[data_locations]

        labels = np.array(data_sample['Yards'])

        labels = np.pad(labels, (0, 25 - len(labels)),

                        mode='constant',

                        constant_values=0)



        numerical_features = get_numerical_features(data_sample)

        features = numerical_features



        padding_length = 25 - features.shape[0]

        inds = np.where(np.isnan(features))

        features[inds] = np.take(np.nanmean(features, axis=0), inds[1])

        inds = np.where(np.isnan(features))

        features[inds] = 0

    

        if padding_length != 0:

            padding_values = np.vstack([np.mean(features, axis=0).reshape(1, -1)]*padding_length)

        

        features = np.concatenate((features, padding_values), axis=0)

        return features, labels
train_df = train_df.sample(frac=1).reset_index(drop=True)

split = np.int32(0.8 * len(train_df))



train_set = NFLCompetitionDataset(data=train_df.iloc[:split], stage='train')

val_set = NFLCompetitionDataset(data=train_df.iloc[split:], stage='val')
class CNN1DNetwork(nn.Module):

    

    def __init__(self):

        super(CNN1DNetwork, self).__init__()

        

        self.conv1d_1 = nn.Conv1d(in_channels=17, out_channels=100, kernel_size=2)

        self.conv1d_2 = nn.Conv1d(in_channels=17, out_channels=100, kernel_size=3)

        self.conv1d_3 = nn.Conv1d(in_channels=17, out_channels=100, kernel_size=4)

        self.conv1d_4 = nn.Conv1d(in_channels=17, out_channels=100, kernel_size=5)

        

        self.dense_1 = nn.Linear(in_features=400, out_features=64)

        self.dense_2 = nn.Linear(in_features=64, out_features=25)

        self.relu = nn.ReLU()

        

    def forward(self, x):

        x = x.float().permute(0, 2, 1)

        conv_1 = self.conv1d_1(x)

        conv_2 = self.conv1d_2(x)

        conv_3 = self.conv1d_3(x)

        conv_4 = self.conv1d_4(x)

    

        max_pool_1, _ = torch.max(conv_1, 2)

        max_pool_2, _ = torch.max(conv_2, 2)

        max_pool_3, _ = torch.max(conv_3, 2)

        max_pool_4, _ = torch.max(conv_4, 2)

        

        features = torch.cat((max_pool_1, max_pool_2, max_pool_3, max_pool_4), 1)

        conc = self.dense_1(features)

        conc = self.relu(conc)

        out = self.dense_2(conc)

        return out
hl_graph = hl.build_graph(CNN1DNetwork(), torch.zeros([1, 25, 17]))

hl_graph.theme = hl.graph.THEMES["blue"].copy()

hl_graph
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
mean = 0.

std = 0.

nb_samples = 0.



for data, _ in tqdm(train_loader):

    batch_samples = data.size(0)

    data = data.view(batch_samples, data.size(1), -1)

    mean += data.mean((0, 1))

    std += data.std((0, 1))

    nb_samples += batch_samples



mean /= nb_samples

std /= nb_samples
start = time.time()

network = CNN1DNetwork()

optimizer = Adam(network.parameters(), lr=0.01)

train_losses = []

val_losses = []



for epoch in range(EPOCHS):

    print("EPOCH " + str(epoch + 1))

    print("")

    

    for (train_batch, val_batch) in zip(train_loader, val_loader):

        train_X, train_y = train_batch

        val_X, val_y = val_batch

        

        train_len = train_X.shape[0]

        val_len = val_X.shape[0]

        

        train_mean = torch.cat(train_len*[torch.cat(25*[mean.view(1, 17)], 0).view(1, 25, 17)], 0)*train_len

        train_std = torch.cat(train_len*[torch.cat(25*[std.view(1, 17)], 0).view(1, 25, 17)], 0)*train_len

        val_mean = torch.cat(val_len*[torch.cat(25*[mean.view(1, 17)], 0).view(1, 25, 17)], 0)*val_len

        val_std = torch.cat(val_len*[torch.cat(25*[std.view(1, 17)], 0).view(1, 25, 17)], 0)*val_len



        train_X = (train_X - train_mean)/train_std

        val_X = (val_X - val_mean)/val_std



        train_y = torch.tensor(train_y, dtype=torch.float)

        val_y = torch.tensor(val_y, dtype=torch.float)

        

        train_preds = network.forward(train_X)

        train_loss = nn.MSELoss()(train_preds, train_y)

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        

        val_preds = network.forward(val_X)

        val_loss = nn.MSELoss()(val_preds, val_y)

    

    end = time.time()

    

    train_losses.append(train_loss.item())

    val_losses.append(val_loss.item())



    print("Train loss: " + str(np.round(train_loss.item(), 3)) + "   " +\

          "Val loss: " + str(np.round(val_loss.item(), 3)) + "   " +\

          "Total time: " + str(np.round(end - start, 1)) + " s")

    print("")
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5, 6, 7, 8], y=train_losses,

    name='train', mode='lines+markers',

    marker_color='crimson'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5, 6, 7, 8], y=val_losses,

    name='val', mode='lines+markers',

    marker_color=' indigo'

))



fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(title='Loss over the epochs', yaxis_zeroline=False, xaxis_zeroline=False)

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="MSE Loss"), xaxis=dict(title="Epochs"))

fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4',

        'Epoch 5', 'Epoch 6', 'Epoch 7', 'Epoch 8']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=train_losses, marker={'color' : 'crimson'}),

    go.Bar(name='val', x=labels, y=val_losses, marker={'color' : 'indigo'})

])



fig.update_layout(title="Loss over the epochs", yaxis=dict(title="MSE Loss"))

fig.update_layout(barmode='group')

fig.show()
test_mean = torch.cat(25*[mean.view(1, 17)], 0).view(1, 25, 17)

test_std = torch.cat(25*[std.view(1, 17)], 0).view(1, 25, 17)
def generate_prediction(data_sample):

    numerical_features = get_numerical_features(data_sample)

    features = numerical_features



    padding_length = 25 - features.shape[0]

    length = features.shape[0]

    inds = np.where(np.isnan(features))

    features[inds] = np.take(np.nanmean(features, axis=0), inds[1])

    inds = np.where(np.isnan(features))

    features[inds] = 0

    

    if padding_length != 0:

        padding_values = np.vstack([np.mean(features, axis=0).reshape(1, -1)]*padding_length)

        

    features = np.concatenate((features, padding_values), axis=0)

    features = (features - test_mean.numpy())/test_std.numpy()

    prediction = network.forward(torch.FloatTensor(features).view(1, 25, 17)).detach().numpy().reshape((25, 1))

    pred = np.zeros((length, 199))



    for index, row in enumerate(prediction):

        if np.int32(np.round(row[0])) < 100:

            pred[index][np.int32(np.round(row[0])) + 99:] = 1

        else:

            pred[index][-1] = 1



        if index == length - 1:

            break



    return pred
for (test_df, sample_prediction_df) in tqdm(env.iter_test()):

    predictions = generate_prediction(test_df)

    env.predict(pd.DataFrame(data=predictions, columns=sample_prediction_df.columns))
env.write_submission_file()
print("Thank you!")