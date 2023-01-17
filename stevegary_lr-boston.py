# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.datasets import load_boston
dir(load_boston())
print(load_boston().DESCR)



# Any results you write to the current directory are saved as output.
X = load_boston().data
y = load_boston().target
import pandas as pd

df = pd.DataFrame(X,columns=load_boston().feature_names)
df.head(10)
df.info()
df.describe().T
#-*- coding: utf-8 -*-  
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus']=False
plt.scatter(df['CRIM'],y)
plt.title('City per capita crime rate and house price scatter chart')
plt.xlabel('Urban per capita crime rate')
plt.ylabel('House price')
plt.show()
def drawScatter(x,y,xlabel):
    plt.scatter(x,y)
    plt.title('%s and house price scatter chart' %xlabel)
    plt.xlabel(xlabel)
    plt.ylabel('house price')
    plt.yticks(range(0,60,5))
    plt.grid()
    plt.show()
drawScatter(df['ZN'],y,'Proportion of residential land')
drawScatter(df['INDUS'],y,'Proportion of non-commercial land in urban areas')
plt.xticks([0,1])
drawScatter(df['CHAS'],y,'Is it on the Charles River?')
drawScatter(df['NOX'],y,'Nitric oxide concentration')
drawScatter(df['RM'],y,'Number of residential rooms')
drawScatter(df['AGE'],y,'Proportion of owner-occupied units built before 1940')
drawScatter(df['DIS'],y,'Average distance from 5 employment centers')
drawScatter(df['RAD'],y,'Convenience index from the expressway')
drawScatter(df['TAX'],y,'Real estate tax rate')
drawScatter(df['PTRATIO'],y,'Student teacher ratio')
drawScatter(df['B'],y,'Black ratio')
rawScatter(df['LSTAT'], y, 'Low-income class')
field_cut = {
    'CRIM' : [0,10,20, 100],
    'ZN' : [-1, 5, 18, 20, 40, 80, 86, 100], 
    'INDUS' : [-1, 7, 15, 23, 40],
    'NOX' : [0, 0.51, 0.6, 0.7, 0.8, 1],
    'RM' : [0, 4, 5, 6, 7, 8, 9],
    'AGE' : [0, 60, 80, 100],
    'DIS' : [0, 2, 6, 14],
    'RAD' : [0, 5, 10, 25],
    'TAX' : [0, 200, 400, 500, 800],
    'PTRATIO' : [0, 14, 20, 23],
    'B' : [0, 100, 350, 450],
    'LSTAT' : [0, 5, 10, 20, 40]
}

cut_df = pd.DataFrame()
for field in field_cut.keys():
    cut_series = pd.cut(df[field], field_cut[field], right=True)
    onehot_df = pd.get_dummies(cut_series, prefix=field)
    cut_df = pd.concat([cut_df, onehot_df], axis=1)
new_df = pd.concat([df, cut_df], axis=1)
new_df.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

X = new_df.values
score_list = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    train_X = X[train_index]
    test_X = X[test_index]
    train_y = y[train_index]  
    test_y = y[test_index]
    linear_model = LinearRegression()
    linear_model.fit(train_X, train_y)
    score = linear_model.score(test_X, test_y)
    score_list.append(score)
    print(score)
np.mean(score_list)