# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import random

Week_days = ['Monday', 'Tuesday','Wednesday','Thursday','Friday'] 
Weekend = ['Saturday','Sunday']
Hour = []
Dates = []
nb_passengers = []
weekday_yes = []

var = 0
i = -1

while var<100:  
    if i < len(Week_days)-1:
        i+=1
    else : 
        i = 0
    for j in range(15): 
        Dates.append(Week_days[i])
        weekday_yes.append(1)
        Hour.append(j+6)
        if (j+6) in [7,8,17,18,19,20]: 
            nb_passengers.append(random.randint(100,200))
        else: 
            nb_passengers.append(random.randint(50,100))
    var += 1


var = 0
i = -1
while var<100:  
    if i < len(Weekend)-1:
        i+=1
    else:
        i=0
    for j in range(15): 
        Dates.append(Weekend[i])
        weekday_yes.append(0)
        Hour.append(j+6)
        if (j+6) in [7,8,17,18,19,20]: 
            nb_passengers.append(round(random.randint(100,200)/2))
        else: 
            nb_passengers.append(round(random.randint(50,100)/2))
    var += 1

df = list(zip(Hour,nb_passengers,weekday_yes,Dates))
data = pd.DataFrame(df, columns=["Hour", "nb_passengers", "weekday_yes","Dates"])

data.shape
data.head(15)
from sklearn.utils import shuffle
import random
random.seed(40)
data=shuffle(data)
data.to_csv('fakedata_bus.csv', header=True)
data.head(20)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
x = data.iloc[:,0]
y = data.iloc[:,1]
plt.scatter(x, y)
plt.xlabel('hours')
plt.ylabel('passengers');
X = data.drop("nb_passengers", axis=1)
Y = data["nb_passengers"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.3)
X_train.head()

cat_features = []
dense_features = []
for col in X_train.columns:
    if X_train[col].dtype =='object':
        cat_features.append(col)
        print("*cat*", col, len(X_train[col].unique()))
    else:
        dense_features.append(col)
        print("!dense!", col, len(X_train[col].unique()))


import tqdm
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
# Encoding categorical data
train_cat = X_train[cat_features]
categories = []
most_appear_each_categories = {}
for col in tqdm_notebook(train_cat.columns):
    train_cat.loc[:,col] = train_cat[col].fillna("nan")
    train_cat.loc[:,col] = col + "__" + train_cat[col].astype(str)
    most_appear_each_categories[col] = list(train_cat[col].value_counts().index)[0]
    categories.append(train_cat[col].unique())
categories = np.hstack(categories)
print(len(categories))
params = {'lambda_l1': 0.001, 'lambda_l2': 0.001,
 'num_leaves': 40, 'feature_fraction': 0.4,
 'subsample': 0.4, 'min_child_samples': 10,
 'learning_rate': 0.01,
 'num_iterations': 100, 'random_state': 42}
from lightgbm import LGBMClassifier
class MultiLGBMClassifier():
    def __init__(self, resolution, params):
        ## smoothing size
        self.resolution = resolution
        ## initiarize models
        self.models = [LGBMClassifier(**params) for _ in range(resolution)]
        
    def fit(self, x, y):
        self.classes_list = []
        for k in tqdm_notebook(range(self.resolution)):
            ## train each model
            self.models[k].fit(x, (y + k) // self.resolution)
            ## (0,1,2,3,4,5,6,7,8,9) -> (0,0,0,0,0,1,1,1,1,1) -> (0,5)
            classes = np.sort(list(set((y + k) // self.resolution))) * self.resolution - k
            classes = np.append(classes, 999)
            self.classes_list.append(classes)
            
    def predict(self, x):
        pred199_list = []
        for k in range(self.resolution):
            preds = self.models[k].predict_proba(x)
            classes = self.classes_list[k]
            pred199s = self.get_pred199(preds, classes)
            pred199_list.append(pred199s)
        self.pred199_list = pred199_list
        pred199_ens = np.mean(np.stack(pred199_list), axis = 0)
        return pred199_ens
    
    def _get_pred199(self, p, classes):
        ## categorical prediction -> predicted distribution whose length is 199
        pred199 = np.zeros(199)
        for k in range(len(p)):
            pred199[classes[k] + 99 : classes[k+1] + 99] = p[k]
        return pred199

    def get_pred199(self, preds, classes):
        pred199s = []
        for p in preds:
            pred199 = np.cumsum(self._get_pred199(p, classes))
            pred199 = pred199/np.max(pred199)
            pred199s.append(pred199)
        return np.vstack(pred199s)
def make_pred(test, sample, env, model):
    test = preprocess(test)
    test = drop(test)
    test = test.drop(un_use_features, axis = 1)
    
    ### categorical
    test_cat = test[cat_features]
    for col in (test_cat.columns):
        test_cat.loc[:,col] = test_cat[col].fillna("nan")
        test_cat.loc[:,col] = col + "__" + test_cat[col].astype(str)
        isnan = ~test_cat.loc[:,col].isin(categories)
        if np.sum(isnan) > 0:
#             print("------")
#             print("test have unseen label : col")
            if not ((col + "__nan") in categories):
#                 print("not nan in train : ", col)
                test_cat.loc[isnan,col] = most_appear_each_categories[col]
            else:
#                 print("nan seen in train : ", col)
                test_cat.loc[isnan,col] = col + "__nan"
    for col in (test_cat.columns):
        test_cat.loc[:, col] = le.transform(test_cat[col])

    ### dense
    test_dense = test[dense_features]
    for col in (test_dense.columns):
        test_dense.loc[:, col] = test_dense[col].fillna(medians[col])
        test_dense.loc[:, col] = sss[col].transform(test_dense[col].values[:,None])

    ### divide
    test_dense_players = [test_dense[dense_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]
    test_dense_players = np.stack([t.values for t in test_dense_players]).transpose(1,0, 2)

    test_dense_game = test_dense[dense_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values
#     test_dense_game = np.hstack([test_dense_game, test_dense[dense_player_features][test_dense["IsRusher"] > 0]])
    
    test_cat_players = [test_cat[cat_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]
    test_cat_players = np.stack([t.values for t in test_cat_players]).transpose(1,0, 2)

    test_cat_game = test_cat[cat_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values
#     test_cat_game = np.hstack([test_cat_game, test_cat[cat_player_features][test_dense["IsRusher"] > 0]])

    test_dense_players = np.reshape(test_dense_players, (len(test_dense_players), -1))
    test_dense = np.hstack([test_dense_players, test_dense_game])
    test_cat_players = np.reshape(test_cat_players, (len(test_cat_players), -1))
    test_cat = np.hstack([test_cat_players, test_cat_game])
    test_x = np.hstack([test_dense, test_cat])

    test_inp = test_x
    
    ## pred
    pred = 0
    for model in models:
        _pred = model.predict(test_inp)
        pred += _pred
    pred /= len(models)
    pred = np.clip(pred, 0, 1)
    env.predict(pd.DataFrame(data=pred,columns=sample.columns))
    return pred





