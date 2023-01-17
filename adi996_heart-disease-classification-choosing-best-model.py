# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import chi2
import plotly.figure_factory as ff
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
df.isnull().sum()
df['target'].value_counts()
fig_dims = (13, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(df.corr(),annot=True , ax=ax)
male = df.loc[df['sex']==0]
female = df.loc[df['sex']==1]
plt.figure(figsize=(10,8))
sns.distplot(male['chol'].loc[male['target']==1],color='r')
sns.distplot(male['chol'].loc[male['target']==0],color='y')
sns.distplot(female['chol'].loc[female['target']==1],color='b')
sns.distplot(female['chol'].loc[female['target']==0],color='g')
plt.title("""Cholestrol levels (Red - Male with target = 1 , Yellow - Male with target = 0
          \nBlue - Female with target = 1 , Yellow - Female with target = 0 """)
plt.figure(figsize=(10,8))
sns.distplot(male['trestbps'].loc[male['target']==1],color='r')
sns.distplot(male['trestbps'].loc[male['target']==0],color='y')
sns.distplot(female['trestbps'].loc[female['target']==1],color='b')
sns.distplot(female['trestbps'].loc[female['target']==0],color='g')
plt.title("""Resting blood pressure(Red - Male with target 1 Yellow - Male with target - 0\n
        Blue - Female with target - 1 Green - Female with target - 0)""")
plt.figure(figsize=(10,8))
sns.distplot(male['thalach'].loc[male['target']==1],color='r')
sns.distplot(male['thalach'].loc[male['target']==0],color='y')
sns.distplot(female['thalach'].loc[female['target']==1],color='b')
sns.distplot(female['thalach'].loc[female['target']==0],color='g')
plt.title("""Maximum heart rate(Red - Male with target 1 Yellow - Male with target - 0\n
        Blue - Female with target - 1 Green - Female with target - 0)""")
plt.figure(figsize=(10,8))
sns.distplot(male['age'].loc[male['target']==1],color='r')
sns.distplot(male['age'].loc[male['target']==0],color='y')
sns.distplot(female['age'].loc[female['target']==1],color='b')
sns.distplot(female['age'].loc[female['target']==0],color='g')
plt.title("""Age distribution (Red - Male with target 1 Yellow - Male with target - 0\n
        Blue - Female with target - 1 Green - Female with target - 0)""")
plt.figure(figsize=(10,8))
sns.distplot(male['oldpeak'].loc[male['target']==1],color='r')
sns.distplot(male['oldpeak'].loc[male['target']==0],color='y')
sns.distplot(female['oldpeak'].loc[female['target']==1],color='b')
sns.distplot(female['oldpeak'].loc[female['target']==0],color='g')
plt.title("""Old peak distribution (Red - Male with target 1 Yellow - Male with target - 0\n
        Blue - Female with target - 1 Green - Female with target - 0)""")
plt.figure(figsize=(10,8))
sns.boxplot(x='target',y='oldpeak',hue='sex',data=df,
           palette='Set3')
plt.title("Old peak")
plt.figure(figsize=(10,8))
sns.boxplot(x='target',y='thalach',hue='sex',data=df,
           palette='Set3')
plt.title("Maximum heart rate recorded")
plt.figure(figsize=(10,8))
sns.boxplot(x='target',y=df['chol'],hue='sex',data=df,
           palette='Set3')
plt.title("Cholestrol level")
plt.figure(figsize=(10,8))
sns.boxplot(x='cp',y='chol',hue='target',data=df,
           palette='Set3')
plt.title("Chest pain and Chol level")
plt.figure(figsize=(10,8))
sns.boxplot(x='cp',
            y='thalach',
            hue='target',
            data=df,
           palette='Set3')
plt.title("Chest pain and Maximum heart rate recorded")
plt.figure(figsize=(10,8))
sns.pointplot(x='cp',y='thalach',hue='target',data=df)
plt.figure(figsize=(10,8))
sns.pointplot(x='restecg',y='thalach',hue='target',data=df)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
feature = df.drop(['target'],axis=1) 
target =  df['target']
x_train , x_test , y_train ,y_test = train_test_split(feature,target,test_size=0.2,random_state=42)
print (f"""X_train :{x_train.shape}
X_TEST : {x_test.shape}
Y_TRAIN : {y_train.shape}
Y_TEST : {y_test.shape}""")
lr = LogisticRegression()
lr.fit(x_train,y_train)
print (classification_report(y_test,lr.predict(x_test)))
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train = scaler.fit_transform(x_train)
lr.fit(scaled_train,y_train)
print(classification_report(y_test,lr.predict(scaler.fit_transform(x_test))))
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print (classification_report(y_test,rf.predict(x_test)))
rf = RandomForestClassifier(criterion='entropy')
rf.fit(scaled_train,y_train)
print(classification_report(y_test,rf.predict(scaler.fit_transform(x_test))))

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criteria = ['entropy','gini']
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
                'criterion':criteria}
rscv = RandomizedSearchCV(estimator = rfc,param_distributions=random_grid, n_iter = 100 , cv=3 , verbose = 2 , random_state = 42,
                         n_jobs=-1)
rscv.fit(x_train,y_train)
rscv.best_params_
print(classification_report(y_test,rscv.predict(x_test)))
from sklearn.ensemble import GradientBoostingClassifier
gbc  = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
print (classification_report(y_test,gbc.predict(x_test)))
dc = DecisionTreeClassifier(criterion='entropy')
dc.fit(x_train,y_train)
print (classification_report(y_test,dc.predict(x_test)))