# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.model_selection import cross_val_score

%pylab inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head(10)
train.info()
age = list(train['Age'])

#remove Nan values

age = list(filter(lambda x : x==x ,age))

plt.hist(age, bins=20, label=('countty','df'))

#plt.xlabel('Age')

#plt.ylabel('Count')

plt.show()
sns.countplot(train['Survived'],label="Count")
sns.countplot(train['Pclass'],label="Count")
train = train.drop("Name", axis=1)

train = train.drop("Ticket", axis=1)

train = train.drop("Cabin", axis=1)
train = train.fillna(train.mean())
train = train.dropna()
train.info()
train = pd.get_dummies(train)
target = train['Survived']
train = train.drop("Survived", axis=1)
model = LinearRegression()

score = mean(sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = 5)))

print("linear regression score: ", score)
cv = 5 #number of folds in cross-validation



#created an array with (10^-5,10^2,with 20 floats in between)

alphas = np.logspace(-8,8,20)



#created a mutli-dimensional array with len(alpha) or 20 of arrays with cv or 5 zeros in each

scores = np.zeros((len(alphas),cv))



#created mu and sigma both an array of 20 zeros

scores_mu = np.zeros(len(alphas))

scores_sigma = np.zeros(len(alphas))



for i in range(0,len(alphas)):

    model = Ridge(alpha=alphas[i])

    scores[i,:] = sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))

    scores_mu[i] = mean(scores[i,:])

    scores_sigma[i] = std(scores[i,:])



figure(figsize(8,4))   

#for i in range(0,cv):

#    plot(alphas,scores[:,i], 'b--', alpha=0.5)

plot(alphas,scores_mu,'c-',lw=3, alpha=0.5, label = "Ridge")

fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),

             np.array(scores_mu)+np.array(scores_sigma),color="c",alpha=0.5)



print("best score in Ridge: ",min(scores_mu))



for i in range(0,len(alphas)):

    model = Lasso(alpha=alphas[i])

    scores[i,:] = sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))

    scores_mu[i] = mean(scores[i,:])

    scores_sigma[i] = std(scores[i,:])



plot(alphas,scores_mu,'g-',lw=3, alpha=0.5, label="Lasso")

fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),

             np.array(scores_mu)+np.array(scores_sigma),color="g",alpha=0.5)



xscale("log")

plt.xlabel("alpha", size=20)

plt.ylabel("rmse", size=20)

legend(loc=2)



print("best score in Lasso: ",min(scores_mu))