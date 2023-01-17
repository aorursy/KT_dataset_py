# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

%matplotlib inline



# We dont Probably need the Gridlines. Do we? If yes comment this line

sns.set(style="ticks")



flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

flatui = sns.color_palette(flatui)

# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

import scipy.stats as ss

from collections import Counter

import math 

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from scipy import stats

import numpy as np
player_df = pd.read_csv("../input/data.csv")
numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']

catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
player_df = player_df[numcols+catcols]
traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)

features = traindf.columns



traindf = traindf.dropna()
traindf = pd.DataFrame(traindf,columns=features)
y = traindf['Overall']>=80

y=y.apply(lambda x : 1 if x else 0).values



X = traindf.copy()

del X['Overall']
X.head()
len(X.columns)
from sklearn.ensemble import RandomForestClassifier

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(X, y)
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model,n_iter=2).fit(X, y)
import eli5

eli5.show_weights(perm, feature_names = X.columns.tolist())
import numpy as np



from lightgbm import LGBMClassifier



lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,

            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)



lgbc.fit(X,y)
from sklearn.metrics import accuracy_score

#define a score function. In this case I use accuracy

def score(X, y):

    y_pred = lgbc.predict(X)

    return accuracy_score(y, y_pred)
from eli5.permutation_importance import get_score_importances

# This function takes only numpy arrays as inputs

base_score, score_decreases = get_score_importances(score, np.array(X), y)

feature_importances = np.mean(score_decreases, axis=0)
feature_importance_dict = {}

for i, feature_name in enumerate(X.columns):

    feature_importance_dict[feature_name]=feature_importances[i]
dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])