import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split 

from ggplot import *
df = pd.read_csv("../input/HR_comma_sep.csv")
df.info()
df.head()
sns.pairplot(df)
cols = ['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',

       'promotion_last_5years']
fig = plt.figure(figsize=(20,12))

corr = df[cols].corr()

sns.heatmap(corr,cbar= True, annot = True, fmt = '.2f', annot_kws = {'size':10}, yticklabels =cols, xticklabels = cols )

plt.show()


ggplot(df, aes(x = 'last_evaluation', y = 'average_montly_hours',size = 'number_project',color = 'salary'))+geom_point()+facet_grid('left')
X, y = df.iloc[:, 0:7].values, df.iloc[:, 7].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,stratify=y)
X_train
y_train