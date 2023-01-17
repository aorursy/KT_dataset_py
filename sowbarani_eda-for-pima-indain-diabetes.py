# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_roc_curve

from sklearn import metrics

from sklearn.tree import plot_tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.sample(5)
df.describe()
df.isnull().sum()


y = df['Outcome']



x = df.drop('Outcome',axis = 1)
ax = sns.countplot(y,label='Count')                               # Total count = 768 Yes = 500 and No = 268

Y, N = y.value_counts()

print("Number of diabetes : ", Y)

print("Number of non diabetes : ", N)
data_outcome = y

data = x

data_n_2 = (data - data.mean())/(data.std())



data = pd.concat([y,data_n_2],axis=1)



data_up = pd.melt(data,id_vars= "Outcome", var_name="features", value_name = 'value')



data_up.head(5)



plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="Outcome", data = data_up, split=True, inner="quart")

plt.xticks(rotation=90);
plt.figure(figsize=(10,10))

sns.boxplot(x="features",y="value",hue='Outcome',data= data_up)

plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

sns.swarmplot(x="features",y="value",hue="Outcome",data = data_up)

plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(x.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)



lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()
lr.fit(x_train,y_train);

dt.fit(x_train,y_train);

rf.fit(x_train,y_train);
lr.score(x_test,y_test)
disp = plot_roc_curve(lr,x_test,y_test);

plot_roc_curve(dt,x_test,y_test);

plot_roc_curve(rf,x_test,y_test);



plt.figure(figsize=(25,10))

a = plot_tree(dt, 

              feature_names=None,

              class_names=None, 

              filled=True, 

              rounded=True, 

              fontsize=14)
y_pred = dt.predict(x_test)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred))
dt_cl1 = DecisionTreeClassifier(criterion='entropy',max_depth=2)
dt_cl1.fit(x_train,y_train)
y_pred_cl1 = dt_cl1.predict(x_test)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred_cl1))


plt.figure(figsize=(25,10))

a = plot_tree(dt_cl1, 

              feature_names=None,

              class_names=None, 

              filled=True, 

              rounded=True, 

              fontsize=14)
print('Accuracy :',metrics.accuracy_score(y_test,y_pred_cl1))