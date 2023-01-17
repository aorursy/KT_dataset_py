# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import Lasso

from scipy.signal import argrelextrema

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error as MAE

import seaborn as sns



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",

                    dtype={'acoustic_data': np.int16

                           , 'time_to_failure': np.float64

                          },engine='c',low_memory=True)
train
def split_to_expirements(t):

    def split(t): # split to matrix that represent the samples and coresponding y

        n = len(t)

        n = n - n % 4096

        x = np.array(t["acoustic_data"].values[:n]).reshape([-1,4096])

        y = np.array(t["time_to_failure"][4095:n:4096])

        return x,y

    x,y = split(t) 

    quake_point = [i+1 for i in argrelextrema(y, np.less, order=2)[0]] # get split indexes by quake times

    return [(x,y) for x,y in zip(np.split(x, quake_point, axis=0), np.split(y,quake_point,axis=0))]
exps = split_to_expirements(train)

del train
def create_train_data(exps):

    res_x = []

    res_y = []

    for x,y in exps:

        n = len(y)

        n = n - (n % 36)

        x = x[:n]

        y = y[:n]

        res_x.extend(np.split(x, n//36, axis=0))

        res_y.extend(y[35:n:36])

    return res_x, res_y
train_x, train_y = create_train_data(exps)

del exps
def filter(train_x, train_y):

    train_x = np.array(train_x)

    train_y = np.array(train_y)

    mask = np.logical_and(train_y <= 12.5,train_y >=1)

    return train_x[mask], train_y[mask]
train_x_filtered, train_y_filtered = filter(train_x, train_y)
X_train, X_test, y_train, y_test = train_test_split(train_x_filtered, train_y_filtered, test_size=0.33, random_state=42)
def preprocess(x):

    quant = [0,0.01,0.05,0.95,0.99,1]

    res = []

    

    for sample in x:

        abs_sample = np.abs(sample)

        features = []

        

        features.append(np.max(abs_sample))

        features.append(np.mean([np.max(ms) for ms in abs_sample]))

        

        features.append(np.std(sample)**2)

        features.append(np.std(abs_sample)**2)

        features.append(np.mean([np.std(ms)**2 for ms in sample]))

        features.append(np.mean([np.std(ms)**2 for ms in abs_sample]))

        

#         for ms in sample:

            

        

        res.append(np.array(features))

    

    return np.array(res)
post_X_test = preprocess(X_test)

post_X_train = preprocess(X_train)
train_test_err = []

alphas = np.logspace(-3,5,num=20)

for alpha in alphas:

    model = Lasso(alpha=alpha)

    model.fit(post_X_train, y_train)

    y_pred_train = model.predict(post_X_train)

    y_pred_test = model.predict(post_X_test)

    train_test_err.append([MAE(y_train, y_pred_train), MAE(y_test, y_pred_test)])

train_test_err = np.array(train_test_err)
plt.plot( alphas, train_test_err[:,0], label='train')

plt.plot( alphas, train_test_err[:,1], label='test')

plt.legend()

plt.xscale('log')
best_alpha = alphas[np.argmin(train_test_err[:,1])]

best_alpha = 1.2
model = Lasso(alpha=best_alpha)

model.fit(post_X_train, y_train)

y_pred_train = model.predict(post_X_train)

y_pred_test = model.predict(post_X_test)

print("train:", MAE(y_train, y_pred_train), "test:" ,MAE(y_test, y_pred_test))
np.set_printoptions(precision=3)

print(model.intercept_)

# print(model.coef_.reshape(37,-1))

print(model.coef_)
y_test = np.array(y_test)

err = np.abs(y_test-y_pred_test)

print(np.mean(err[y_test<10]))

sns.distplot(err[y_test<10])
def to_segment(X_test):

    n = len(X_test)

    n = n - (n % 4096)

    return X_test[:n].reshape(-1,4096)
subm = pd.read_csv('../input/sample_submission.csv')
def pred(fn):

    test = pd.read_csv(f"../input/test/{fn}.csv",

                    dtype={'acoustic_data': np.int16},engine='c',low_memory=True)

    test = np.array(test)

    test=to_segment(test).reshape([1,36,4096])

    test = preprocess(test)

    return model.predict(test)[0]
subm['time_to_failure'] = [pred(x) for x in subm['seg_id'].values]
max_with_0 = subm['time_to_failure'].values

max_with_0[max_with_0<0]=0

subm['time_to_failure']=max_with_0
subm.to_csv("submission.csv", index=False)
subm_1 = pd.read_csv('submission.csv')
np.allclose(subm_1['time_to_failure'].values,subm['time_to_failure'].values)
subm_1