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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # XGBoost implementation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# read data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



features = [x for x in train.columns if x not in ['id','loss']]

#print(features)



cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]

#print(cat_features)

#print(num_features)

from scipy.stats import norm, lognorm

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



train['log_loss'] = np.log(train['loss'])



# fit the normal distribution on ln(loss)

(mu, sigma) = norm.fit(train['log_loss'])



# the histogram of the ln(loss)

n, bins, patches = plt.hist(train['log_loss'], 60, normed=1, facecolor='green', alpha=0.75)



# add the fitted line

y = mlab.normpdf( bins, mu, sigma)

l = plt.plot(bins, y, 'r--', linewidth=2)



#plot

plt.xlabel('Ln(loss)')

plt.ylabel('Probability')

plt.title(r'$\mathrm{Histogram\ of\ Ln(Loss):}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

plt.grid(True)



plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

train_test = pd.concat((train[features], test[features])).reset_index(drop=True)

for c in range(len(cat_features)):

    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes



train_x = train_test.iloc[:ntrain,:]

test_x = train_test.iloc[ntrain:,:]

xgdmat = xgb.DMatrix(train_x, train['log_loss'])

print(xgdmat)



params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 

             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 



# Grid Search CV optimized settings

num_rounds = 1000

bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)

print(bst)