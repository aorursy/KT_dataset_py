import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from time import time
df = pd.read_csv('../input/HR_comma_sep.csv')
df.info()
df.head()
df['salary'].unique()
df.groupby('left').count()['salary'].plot(kind='bar', color='g', title='Stayed "0" VS. Left "1"', width =.2,stacked = True)

print('The Number of Employees Left = {} \n Total number of Employees = {}'.format(df[df['left']==1].shape[0], df.shape[0]))
order = ['low', 'medium', 'high']

df.groupby('salary').count()['sales'].loc[order].plot(kind='bar', color='rgb', title='The Percentile of each Salary Class', width =.2,stacked = True)

print('The Number of Employees with low salary = {} \n The Number of Employees with medium salary = {} \n The Number of Employees with high salary = {}'.format(df[df['salary']=='low'].shape[0], df[df['salary']=='medium'].shape[0], df[df['salary']=='high'].shape[0]))
df_low = df[df['salary'] == 'low']

df_medium = df[df['salary'] == 'medium']

df_high = df[df['salary'] == 'high']



fmt= '{:<25}{:<25}{}'



print(fmt.format('','mean','std'))



for i,(mean,std) in enumerate(zip(df_low.mean() , df_low.std() )):

    print(fmt.format(df_low.columns[i], mean, std))

print('\n')

for i,(mean,std) in enumerate(zip(df_medium.mean() , df_medium.std() )):

    print(fmt.format(df_medium.columns[i], mean, std))

print('\n')

for i,(mean,std) in enumerate(zip(df_high.mean() , df_high.std() )):

    print(fmt.format(df_high.columns[i], mean, std))
corrmat = df.corr()

f, ax =plt.subplots(figsize=(4,4))

# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=0.8, square = True)

plt.show()
df.groupby('sales').mean()['satisfaction_level'].plot(kind='bar', colormap='hot')
df.groupby('sales').mean()['left'].plot(kind='bar', colormap='gist_rainbow')
sns.factorplot("sales", col="salary", col_wrap=3, data=df, kind="count", size=10, aspect=.8)
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
df_copy = pd.get_dummies(df)

df_copy.head()
df1 = df_copy

y = df1['left'].values

df1 = df1.drop(['left'],axis=1)

X = df1.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
log_reg = LogisticRegression()

log_reg.fit(Xtrain, ytrain)

pred_log = log_reg.predict(Xtest)

accuracy = accuracy_score(pred_log, ytest)

print("Validation accuracy: ", accuracy)
clf= svm.SVC(C=10000,kernel="rbf")

t0=time()

clf.fit(Xtrain,ytrain)

print("RBF Kernel, C=10000\ntraining time: ", round(time()-t0, 3), "s")



t1=time()

pred=clf.predict(Xtest)

print("predicting time: ", round(time()-t1, 3),"s")

accuracy= accuracy_score(pred, ytest)

print(accuracy)