# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
features = pd.concat([train.drop(columns=['SalePrice']), test])

labels = train['SalePrice']
features.columns
features[['YearBuilt', 'YearRemodAdd', 'YrSold']].isna().sum()
features[features['YearRemodAdd'] - features['YearBuilt']<0][['YearBuilt', 'YearRemodAdd', 'YrSold']]
features['year_since'] = features['YrSold'] - features['YearRemodAdd']

features['total_area'] = features['TotalBsmtSF'] + features['GrLivArea']



c_feat = features[['year_since', 'total_area']]
c_feat.isna().sum()
features[np.isnan(features['TotalBsmtSF'])][['TotalBsmtSF', 'GrLivArea']]
c_feat = c_feat.fillna(896)

c_feat.isna().sum()
def standard_scaler(x):

    return (x-x.mean())/x.std()



c_feats = c_feat.apply(standard_scaler)

c_feats['constant'] = np.ones(len(train)+len(test))



train = c_feats[:len(train)]

test = c_feats[len(train):]

label = labels



def rmse(yhat, y):

    return np.sqrt(np.sum((yhat-y)**2/len(y)))



c_feats
class linear_regression:

    def __init__(self, lr=0.1):

        self.theta = np.random.normal(0, 1, (1,3))

        self.lr = lr

        

    def regression(self, x):

        return np.dot(self.theta, x.T)

    

    def back_propagation(self, x, y):

        return (self.regression(x) - y) * x

    

    def training(self, x, y):

        self.theta = self.theta - self.lr * self.back_propagation(x, y)

        

linreg = linear_regression(lr=0.1)

rmses = []    

    

epoch = 150

    

for k in range(epoch):

    linreg.lr = linreg.lr * 0.8

    for i, v in train.iterrows():

        linreg.training(np.array(v), label[i])

    temp = linreg.regression(train)

    yhat = [item for sublist in temp for item in sublist]

    y = label

    rmses.append(rmse(yhat, y))
import matplotlib.pyplot as plt



plt.figure(figsize=(16,9))

plt.title("Rmse over iterations")

plt.xlabel('# iterations')

plt.ylabel('RMSE')

plt.plot(rmses, color='royalblue', linewidth=3)

plt.show()
rmses[-1]