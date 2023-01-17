!pip install -U torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from glob import glob
import json
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from tqdm.notebook import tqdm
import random
import torch
import torch.nn as nn
print(torch.__version__)
infos = sorted(glob(r"/kaggle/input/halite-top-games-2/*/*_info.json"))
games = []
for info in infos:
    log = json.load(open(info, 'r'))
    for agent in log['agents']:
        if agent['submissionId'] == 16647790:
            games.append(info[:-10] + info[-5:])
            break
games[:10]
def get_pos(s):
    return (s // 21, s % 21)
print(get_pos(22))

def dry_move(pos, d):
    if type(pos) is int:
        pos = get_pos(pos)
    if d == "NORTH":
        pos = ((pos[0] - 1) % 21, pos[1] % 21)
    elif d == "SOUTH":
        pos = ((pos[0] + 1) % 21, pos[1] % 21)
    elif d == "EAST":
        pos = (pos[0] % 21, (pos[1] + 1) % 21)
    elif d == "WEST":
        pos = (pos[0] % 21, (pos[1] - 1) % 21)
    return pos
print(dry_move(22, "NORTH"))
# clear directory
import shutil
if os.path.isdir("./data"):
    shutil.rmtree("./data")
os.mkdir("./data")

data = []
# preprocess input & output
for game in tqdm(games):
    log = json.load(open(game, 'r'))
    info = json.load(open(game[:-5] + "_info.json", 'r'))
    
    # if more than one mzotkiew, continue
    if np.sum([x['submissionId'] == 16647790 for x in info['agents']]) > 1:
        continue
    
    playerId = np.argmax([x['submissionId'] == 16647790 for x in info['agents']])
    for steplog in log['steps']:
        if steplog[playerId]['action'] is None:
            continue
        # retrieve data
        obs = steplog[0]['observation']
        bank = obs['players'][playerId][0]
        shipCnt = len(obs['players'][playerId][2])
        totalShipCnt = 0
        for player in obs['players']:
            totalShipCnt += len(player[2])
        step = obs['step']
        haliteMean = sum(obs['halite']) / len(obs['halite'])
        
        # isBlocked
        if len(obs['players'][playerId][1]) != 1: # workaround
            continue
        if step >= len(log['steps']) - 1:
            continue
        
        shipyardPos = get_pos(list(obs['players'][playerId][1].values())[0])
        isBlocked = 0
        actions = log['steps'][step + 1][playerId]['action'] # future action
        for k, v in obs['players'][playerId][2].items(): # all ships
            nextPos = dry_move(v[0], actions[k]) if k in actions else get_pos(v[0])
            if nextPos == shipyardPos:
                isBlocked = 1
                
        # label
        label = "SPAWN" in log['steps'][step + 1][playerId]['action'].values()
        
        # save data to memory
        data.append(torch.tensor([bank, totalShipCnt, shipCnt, step, haliteMean, isBlocked, label]))

data = torch.stack(data)
print(data[:, 5].sum())
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
def test(testSet, verbose=False):
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=32)
    tp, tn, fp, fn = 0, 0, 0, 0
    for x, label in testLoader:
        y = model(x).squeeze()
        tp += torch.sum(np.logical_and(y > 0, label == 1)).item()
        tn += torch.sum(np.logical_and(y < 0, label == 0)).item()
        fp += torch.sum(np.logical_and(y > 0, label == 0)).item()
        fn += torch.sum(np.logical_and(y < 0, label == 1)).item()
    tp /= len(testSet)
    tn /= len(testSet)
    fp /= len(testSet)
    fn /= len(testSet)
    if verbose:
        print(tp, fp, tn, fn)
    return (tp + tn) / (tp + tn + fp + fn)
# get mean, std
X, Y = data[:, :6], data[:, 6]
X = (X - X.mean(0)) / X.std(0)
print(X.shape, Y.shape)
trainSet, testSet = torch.utils.data.random_split(torch.utils.data.TensorDataset(X, Y), [len(X) - len(X) // 5, len(X) // 5])
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32)
lossFn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001) # I trained many times with decreasing lr
    
# train
epochs = 100
for epoch in tqdm(range(epochs)):
    runningLoss = 0.
    for x, label in trainLoader:
        y = model(x).squeeze()
        loss = lossFn(y, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        runningLoss += loss.item()
    if epoch % (epochs // 10) == 0:
        print(f"epoch{epoch + 1}/{epochs}, loss={runningLoss}, train_acc={test(trainSet)}, test_acc={test(testSet)}")
test(testSet, True)
sns.scatterplot(x=range(100), y=model(testSet[:100][0]).detach().numpy().squeeze())
for param in model.parameters():
    print(param.detach().numpy())
