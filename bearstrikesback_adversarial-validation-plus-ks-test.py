#Load packages

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
#Load data; drop target and ID's

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.drop(train[['ID_code', 'target']], axis=1, inplace=True)

test.drop(test[['ID_code']], axis=1, inplace=True)
#Create label array and complete dataset

y1 = np.array([0]*train.shape[0])

y2 = np.array([1]*test.shape[0])

y = np.concatenate((y1, y2))



X_data = pd.concat([train, test])

X_data.reset_index(drop=True, inplace=True)
#Initialize splits&LGBM

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)



lgb_model = lgb.LGBMClassifier(max_depth=-1,

                                   n_estimators=100,

                                   learning_rate=0.1,

                                   objective='binary', 

                                   n_jobs=-1)

                                   

counter = 1
#Train 5-fold adversarial validation classifier

for train_index, test_index in skf.split(X_data, y):

    print('\nFold {}'.format(counter))

    X_fit, X_val = X_data.loc[train_index], X_data.loc[test_index]

    y_fit, y_val = y[train_index], y[test_index]

    

    lgb_model.fit(X_fit, y_fit, eval_metric='auc', 

              eval_set=[(X_val, y_val)], 

              verbose=100, early_stopping_rounds=10)

    counter+=1
#Load more packages

from scipy.stats import ks_2samp

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
#Perform KS-Test for each feature from train/test. Draw its distribution. Count features based on statistics.

#Plots are hidden. If you'd like to look at them - press "Output" button.

hypothesisnotrejected = []

hypothesisrejected = []



for col in train.columns:

    statistic, pvalue = ks_2samp(train[col], test[col])

    if pvalue>=statistic:

        hypothesisnotrejected.append(col)

    if pvalue<statistic:

        hypothesisrejected.append(col)

        

    plt.figure(figsize=(8,4))

    plt.title("Kolmogorov-Smirnov test for train/test\n"

              "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, statistic, pvalue))

    sns.kdeplot(train[col], color='blue', shade=True, label='Train')

    sns.kdeplot(test[col], color='green', shade=True, label='Test')



    plt.show()
len(hypothesisnotrejected), len(hypothesisrejected)
print(hypothesisrejected)