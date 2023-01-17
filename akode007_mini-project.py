import numpy as np

import pandas as pd

import os

import math

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)

from sklearn.metrics import accuracy_score

from scipy.stats import norm,skew

from sklearn import metrics
data = pd.read_csv('../input/zoo.csv')

df2 = pd.read_csv('../input/class.csv')

data.head()
df2
# Check data type for each variable

data.info()
#lets try to assess missing values

data.isnull().sum()
data.describe()
print(data.shape, df2.shape)
# lets try to join both sets to show actual class names

df=pd.merge(data,df2,how='left',left_on='class_type',right_on='Class_Number')

df.head()

plt.hist(df.class_type, bins=7)
# lets see which class the most zoo animals belong to

sns.factorplot('Class_Type', data=df,kind="count", aspect=2)
corr = df2.corr()

sns.heatmap(corr, square=True, linewidths=.3,cmap="RdBu_r")

plt.show()
corr = data.corr()



sns.heatmap(corr, square=True, linewidths=.2,cmap="RdBu_r")

plt.show()
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("Correlation Heatmap")

corr = data.corr()

sns.heatmap(corr, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# show vairable correlation which is more than 0.7 (positive or negative)

corr_filt = corr[corr != 1][abs(corr)> 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)

print(corr_filt)
df.groupby('Class_Type').mean()
from sklearn.model_selection import train_test_split

# 80/20 split

X = data.iloc[:,1:17]

y = data.iloc[:,17]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print(X.shape,y.shape)
from sklearn.model_selection import cross_val_score

from sklearn import tree

dt = tree.DecisionTreeClassifier()

score_dt=cross_val_score(dt, X,y, cv=5)

score_dt

# The mean score and the 95% confidence interval of the score estimate are:

#print("Accuracy: %0.2f (+/- %0.2f)" % (score_dt.mean(), score_dt.std() * 2))
print("accuracy:" +str(score_dt.mean()))


from sklearn.svm import SVC

svc = SVC(kernel='linear', C=1)

score_svc=cross_val_score(svc, X,y, cv=5)

score_svc
print("accuracy:" +str(score_svc.mean()))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')

score_lr=cross_val_score(lr, X,y, cv=5)

score_lr

#print('Accuracy:', round(score_lr, 2), '%.')

print("accuracy:" +str(score_lr.mean()))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 35,max_depth=7, random_state = 42)

score_rf=cross_val_score(rf,X,y, cv=5)

score_rf

print("accuracy:" +str(score_rf.mean()))




models = pd.DataFrame({

    'Model': ['Support Vector Machines',  'Decision Tree', 'Logistic Regression','Random Forest'],

    'Score': [score_svc.mean(), score_dt.mean(), score_lr.mean(), score_rf.mean()]})

models.sort_values(by='Score', ascending=False)



Model= ['Support Vector Machines',  'Decision Tree', 'Logistic Regression','Random Forest']

Score = [score_svc.mean(), score_dt.mean(), score_lr.mean(), score_rf.mean()]
print(Score)
# Performance of models

fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(6, 2.5))

sns.barplot(Model, Score,palette="RdBu")

plt.ylim(0.94, 0.97)

ax.set_ylabel("Performance")

ax.set_xlabel("Name")

ax.set_xticklabels(Model,rotation=35)

plt.title('Model')
