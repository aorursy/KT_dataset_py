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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape, test.shape
features = [x for x in train.columns if x not in ['id', 'loss']]

cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id', 'loss']]
from scipy.stats import norm, lognorm

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns

train.loss.hist()
np.log(train.loss).hist()
train['log_loss'] = np.log(train.loss)

#fit the normal distribution on ln(loss)

(mu, sigma) = norm.fit(train.log_loss)

#the histogram of the ln(loss)

n, bins, patches = plt.hist(train.log_loss, 60, normed=1)

# add the fitted line

y = mlab.normpdf(bins, mu, sigma)

l = plt.plot(bins, y, 'r--', linewidth=2)

#plot

plt.xlabel('Ln(loss)')

plt.ylabel('Probability')

plt.title(r'$\mathrm{Histogram\ of\ Ln(Loss):}\ \mu=%.3f, \ \sigma=%.3f$' %(mu, sigma))
ntrain = train.shape[0]

ntest = test.shape[0]

train_test = pd.concat((train[features], test[features])).reset_index(drop=True)

for c in range(len(cat_features)):

    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

train_x = train_test.iloc[:ntrain, :]

test_x = train_test.iloc[ntrain:, :]

train_x.head()
train_x.shape, test_x.shape
train.log_loss.head()
import xgboost as xgb

xgdmat = xgb.DMatrix(train_x, train.log_loss)

params = {'eta': 0.01, 'seed': 0, 'subsample': 0.5, 'colsample':0.5,

           'objective':'reg:linear', 'max_depth':6, 'min_child_weight':3}

# Grid search CV optimized settings

num_rounds = 1000

bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
import operator

def create_feature_map(features):

    outfile = open('xgb.fmap', 'w')

    i = 0

    for feat in features:

        outfile.write('{0}\t{1}\tq\n'.format(i, feat))

        i +=1

    outfile.close()
create_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')

#importance = bst.get_fscore()

importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])

df['fscore'] = df['fscore']/df['fscore'].sum()
len(importance), len(features)
plt.figure()

df.plot()

df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

plt.title('XGBoost Feature Importance')

plt.xlabel('relative importance')

plt.gcf().savefig('feature_importance_xgb.png')

df
test_xgb = xgb.DMatrix(test_x)

submission = pd.read_csv('../input/sample_submission.csv')

pred = np.exp(bst.predict(test_xgb))

submission.iloc[:,1] = pred

submission.to_csv('xgb_starter_sub.csv', index=None)