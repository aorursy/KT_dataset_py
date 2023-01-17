import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head(5)
data.info()
data.isnull().sum().sort_values(ascending=False)
#The data are already mapped for us already

#sex gender = male is 1 and female is 0

#diabetes = 1 is yes and 0 is no

#anaemia = 1 is yes and 0 is no

#high blood pressure = 1 is yes and 0 is no

#smoking = 1 is yes and 0 is no

#death events = 1 is yes and 0 is no
#Let us find the distribution plot for ages

f, ax = plt.subplots(figsize=(12,5))

sns.distplot(data['age'], color='g')
#FROM THE DISTPLOT IT SEEMS THAT PEOPLE AROUND THE AGES OF 40 - 70 ARE AFFECTED MORE THAN 

#ANY OTHER AGES IN THE DATASET
df = data.copy()

colors = ["#0101DF", "#DF0101"]

f, ax = plt.subplots(figsize=(12,5))

sns.countplot('sex', data=df, palette=colors )
#WE HAVE MORE MALES THAN FEMALES IN THE DATASET
df = data.copy()

colors = ["#0101DF", "#DF0101"]

f, ax = plt.subplots(figsize=(12,5))

sns.countplot('diabetes', data=df, palette=colors )
f, ax = plt.subplots(figsize=(12,5))

sns.countplot('smoking', data=df)
male = data[data.sex==1]

female = data[data.sex==0]
from plotly.offline import init_notebook_mode,iplot
ds = df['DEATH_EVENT'].value_counts().reset_index()

ds.columns = ['DEATH_EVENT', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="DEATH_EVENT", 

    title='DEATH_EVENT chart', 

    width=600, 

    height=500

)

fig.show()
ds = df['diabetes'].value_counts().reset_index()

ds.columns = ['diabetes', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="diabetes", 

    title='Diabetes bar chart', 

    width=600, 

    height=500

)

fig.show()
#Below is a heatmap of the correlation of the numerical columns:

correlation_matrix = data.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['DEATH_EVENT'].sort_values(ascending=False)
#There is no linear correlation between any of the input variables and death events
y = data['DEATH_EVENT']

X = data.drop(['DEATH_EVENT'], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
classifier = LogisticRegression()

classifier.fit(x_train, y_train)
training_score = cross_val_score(classifier, x_train, y_train, cv=10)
training_score
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(x_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_
log_reg
log_reg_score = cross_val_score(log_reg, x_train, y_train, cv=10)
log_reg_score.mean()
y_pred = log_reg.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
cm
from sklearn.metrics import classification_report
print('Logistic Regression:')

print(classification_report(y_test, y_pred))
#Using RandomForestClassifier
RF = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=0)

RF.fit(x_train, y_train)
pred = RF.predict(x_test)
pred
training_score = cross_val_score(RF, x_train, y_train, cv=10)
training_score
con_mat = confusion_matrix(pred, y_test)
con_mat
print('Random Forest:')

print(classification_report(y_test, pred))
#PLEASE DONT FORGET GIVE AN UPVOTE