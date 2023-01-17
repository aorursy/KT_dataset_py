# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from matplotlib import pyplot as plt

#ML models
from sklearn.svm import SVR
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def test():
    print('test')

def train_model(x,y):
    #regr = SVR(C=1.0, epsilon=0.1, degree=3)
    #regr = svm.SVC()
    regr = DecisionTreeRegressor(max_depth=10000)
    #regr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,10), random_state=3)
    #regr = BayesianRidge(compute_score=True)
    regr.fit(x, y)
    return regr

def predict(model,x):
    model.predict(x)

def player_models(dataset,playerNames):
    models={}
    for playerName in playerNames:
        player_traindata = dataset.loc[playerName]
        player_x = player_traindata.drop(target+['playerName'],1)
        player_y = player_traindata[target]
        player_x = pd.get_dummies(player_x)
        models[playerName] = train_model(player_x,player_y)
        #print(playerName)
    return models

def action_predict(predict_model,predict_data):
    predict_data = predict_data.drop(target+['playerName'],1)
    predict_data = pd.get_dummies(predict_data)
    return predict_model.predict(predict_data)

def draw_figure(target,predict_result):
    plt.figure(figsize=(30,10))
    xy1=[]
    xy2=[]
    test_y=[]
    pred_y=[]
    for i in range(len(predict_result)):
        if abs(predict_result[i] - target[i]) > 0.1:   
            xy1.append((predict_result[i],target[i]))
        else:
            xy2.append((predict_result[i],target[i]))
    
    xy1.sort(key=lambda x: x[1])
    xy2.sort(key=lambda x: x[1])
    
    xy = xy2+xy1
    
    for xy_element in xy:
        pred_y.append(xy_element[0])
        test_y.append(xy_element[1])
    
    
    plt.plot(test_y, color='blue')
    plt.plot(pred_y, color='red')
    
    plt.show()
dataset_719 = pd.read_csv('../input/user_logs_20180719_dataset.csv')
dataset_720 = pd.read_csv('../input/user_logs_20180720_dataset.csv')
print(dataset_719.columns)
feature_list = ['playerName','action','PodOddsValue','activeUser','round']
target = ['win_rate']
dataset_719 = dataset_719[feature_list+target]
dataset_720 = dataset_720[feature_list+target]

dataset_719 = dataset_719.round({'PodOddsValue': 2, 'win_rate': 2})
dataset_720 = dataset_720.round({'PodOddsValue': 2, 'win_rate': 2})
dataset_719.set_index(keys=['playerName'], drop=False,inplace=True)
dataset_720.set_index(keys=['playerName'], drop=False,inplace=True)
playerName_719 = dataset_719['playerName'].unique().tolist()
playerName_720 = dataset_720['playerName'].unique().tolist()
playerName_720_models = {}
playerName_720_models = player_models(dataset_720,playerName_720[0:200])
#models['007216ed0be2fd6ad8ae2211a98b16b1']
#playerName_719_models
for i in range(200):
    playName = playerName_720[i]
    predict_data = dataset_720.loc[playName]
    predict_model = playerName_720_models[playName]

    predict_result = action_predict(predict_model,predict_data)
    predict_data_y = []

    for value in predict_data[target[0]]:
        predict_data_y.append(value)
    draw_figure(predict_data_y,predict_result)