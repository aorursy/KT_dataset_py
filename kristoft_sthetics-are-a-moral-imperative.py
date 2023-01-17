# Let's start off importing some python tools

import pandas as pd

import numpy as np

import random as rnd

from tabulate import tabulate



# Visualization libraries 

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import plotly as py

import plotly.express as px

import plotly.graph_objects as go



#!pip install heatmapz # Really great heatmap visualization from https://pypi.org/project/heatmapz/

#from heatmap import heatmap, corrplot

%matplotlib inline

#%matplotlib notebook



# ML tools

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn import svm

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB,CategoricalNB,MultinomialNB

from sklearn.linear_model import Perceptron, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, accuracy_score

from xgboost import XGBClassifier

from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss



# Importing Datasets to pandas DataFrames

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

complete_df = [train_df, test_df] #Helpful when cleaning both dataframes at once



# Completing or deleting missing values in the dataset

# I am filling in median age, mode embark, and mediam fare for missing values

for datadf in complete_df:    

    #complete missing age with median

    datadf['Age'].fillna(datadf['Age'].median(), inplace = True)



    #complete embarked with mode

    datadf['Embarked'].fillna(datadf['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    datadf['Fare'].fillna(datadf['Fare'].median(), inplace = True)

    

# Getting rid of irrelevant columns: Passanger ID, Cabin number, and Ticker number

drop_column = ['PassengerId','Cabin', 'Ticket']

for datadf in complete_df:    

    datadf.drop(drop_column, axis=1, inplace = True)

    

# Ok, time to visualize the data!

import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

print('x' in np.arange(5))   #returns False, without Warning
# General age and sex distribution with added swarmplot

plt.figure(figsize=(12,8))



# sns plot would add a default legend of 0 or 1 for Survived column, so change it to "Died" and "Survived"

custom = [Line2D([], [], marker='o', color='#023EFF', linestyle='None'),

          Line2D([], [], marker='o', color='#FF7C00', linestyle='None')]

sns.swarmplot(x="Age", y="Sex", hue="Survived", data=train_df, palette="bright")

ax = sns.boxplot(x="Age", y="Sex", data=train_df, color='white')



# Making the boxplot edge colors black for better contrast

for i,box in enumerate(ax.artists):

    box.set_edgecolor('black')

    box.set_facecolor('white')

    # iterate over whiskers and median lines

    for j in range(6*i,6*(i+1)):

         ax.lines[j].set_color('black')

            

# Setting the text for legend, overall font size, and title of figure            

plt.legend(custom, ['Died', 'Survived'], loc='upper right')

plt.rc('font', size=20)

plt.rc('axes', titlesize=20)



# This plot gives a nice view of the phrase "Women and Children first!"
# Let's see how port of embarkation and fare correlate

plt.figure(figsize=(12,8))



# sns plot would add a default legend of 0 or 1 for Survived column, so change it to "Died" and "Survived"

custom = [Line2D([], [], marker='o', color='#023EFF', linestyle='None'),

          Line2D([], [], marker='o', color='#FF7C00', linestyle='None')]

sns.swarmplot(x="Fare", y="Embarked", hue="Survived", data=train_df, palette="bright")

ax = sns.boxplot(x="Fare", y="Embarked", data=train_df, color='white')



# Making the boxplot edge colors black for better contrast

for i,box in enumerate(ax.artists):

    box.set_edgecolor('black')

    box.set_facecolor('white')

    # iterate over whiskers and median lines

    for j in range(6*i,6*(i+1)):

         ax.lines[j].set_color('black')

            

# Setting the text for legend, overall font size, and title of figure            

plt.legend(custom, ['Died', 'Survived'], loc='upper right')

plt.rc('font', size=20)

plt.rc('axes', titlesize=20)
train_df["Survived"] = train_df["Survived"].astype(str)

fig = px.scatter(train_df, x="Age", y="Fare", color="Survived",hover_data=['SibSp','Parch','Pclass','Embarked'],

                 marginal_x="box", marginal_y="box" ,title="Box plots of Age and Fare",hover_name='Name')

fig.update_layout(legend_title_text='Survived? 0=No, 1=Yes')

fig
train_df = train_df.sort_values(by=['Pclass'])

fig = px.scatter(train_df, x="Age", y="Fare",animation_frame="Pclass",

                 color="Embarked", hover_name="Name", facet_col="Survived",

           title="Scatter Plots of Age and Fare",log_x=False, size_max=45)

fig.show()
train_df["Survived"] = train_df["Survived"].astype(str)

train_df = train_df.sort_values(by=['Survived'])

fig = px.scatter_3d(train_df, x='Age', y='Fare', z='Pclass', size = train_df['SibSp']+train_df['Parch'],

              color='Survived', size_max=40,

              symbol='Sex', opacity=0.8,hover_name="Name",hover_data=['SibSp','Parch','Embarked'])

# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# 3D plot based on Age, Fare, and Family size (siblings+spouse+children+parents in z)

train_df["Survived"] = train_df["Survived"].astype(str)

train_df = train_df.sort_values(by=['Survived'])



fig = px.scatter_3d(train_df, x='Age', y='Fare', z=train_df['SibSp']+train_df['Parch'],

              color='Survived', size_max=10,

              symbol='Sex', opacity=0.7,hover_name="Name",hover_data=['SibSp','Parch','Embarked'])

# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
train_df['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

train_df['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)



test_df['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)
X = train_df

y = train_df['Survived'].values

X.drop(['Name'],axis = 1, inplace = True)

X.drop(['Survived'],axis = 1, inplace = True)



X_submit = test_df

X_submit.drop(['Name'],axis = 1, inplace = True)



X = preprocessing.StandardScaler().fit(X).transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# From sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier(criterion="entropy")

DT_model.fit(X_train,y_train)



DT_yhat = DT_model.predict(X_test)



print("DT accuracy: %.2f" % accuracy_score(y_test, DT_yhat))

print("DT Jaccard index: %.2f" % jaccard_score(y_test, DT_yhat,pos_label='1'))

print("DT F1-score: %.2f" % f1_score(y_test, DT_yhat, average='weighted') )
RanFor_model = RandomForestClassifier(n_estimators=10,random_state=1).fit(X_train,y_train)

RanFor_yhat = RanFor_model.predict(X_test)



print("Random Forest accuracy: %.2f" % accuracy_score(y_test, RanFor_yhat))

print("Random Forest Jaccard index: %.2f" % jaccard_score(y_test, RanFor_yhat,pos_label='1'))

print("Random Forest F1-score: %.2f" % f1_score(y_test, RanFor_yhat, average='weighted') )
# From sklearn import svm.SVC()

SVM_model = SVC()

SVM_model.fit(X_train,y_train)



SVM_yhat = SVM_model.predict(X_test)



print("SVM accuracy: %.2f" % accuracy_score(y_test, SVM_yhat))

print("SVM Jaccard index: %.2f" % jaccard_score(y_test, SVM_yhat,pos_label='1'))

print("SVM F1-score: %.2f" % f1_score(y_test, SVM_yhat, average='weighted') )
LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)

LR_yhat = LR_model.predict(X_test)



print("LR accuracy: %.2f" % accuracy_score(y_test, LR_yhat))

print("LR Jaccard index: %.2f" % jaccard_score(y_test, LR_yhat,pos_label='1'))

print("LR F1-score: %.2f" % f1_score(y_test, LR_yhat, average='weighted') )
NB_model = BernoulliNB(2).fit(X_train,y_train)

NB_yhat = NB_model.predict(X_test)



print("NB accuracy: %.2f" % accuracy_score(y_test, NB_yhat))

print("NB Jaccard index: %.2f" % jaccard_score(y_test, NB_yhat,pos_label='1'))

print("NB F1-score: %.2f" % f1_score(y_test, NB_yhat, average='weighted') )
XGB_model=XGBClassifier(max_depth=3).fit(X_train,y_train)

XGB_pred=XGB_model.predict(X_test)

    

print("XGB accuracy: %.2f" % accuracy_score(y_test, XGB_pred))

print("XGB Jaccard index: %.2f" % jaccard_score(y_test, XGB_pred,pos_label='1'))

print("XGB F1-score: %.2f" % f1_score(y_test, XGB_pred, average='weighted') )    

    
from tabulate import tabulate

data = [['Decision Tree', accuracy_score(y_test, DT_yhat), jaccard_score(y_test, DT_yhat,pos_label='1'), f1_score(y_test, DT_yhat, average='weighted')],

['Random Forest Classifier', accuracy_score(y_test, RanFor_yhat), jaccard_score(y_test, RanFor_yhat,pos_label='1'), f1_score(y_test, RanFor_yhat, average='weighted')],

['Support Vector Machine', accuracy_score(y_test, SVM_yhat), jaccard_score(y_test, SVM_yhat,pos_label='1'), f1_score(y_test, SVM_yhat, average='weighted')],

['Logistic Regression', accuracy_score(y_test, LR_yhat), jaccard_score(y_test, LR_yhat,pos_label='1'), f1_score(y_test, LR_yhat, average='weighted')],

['Bernoulli Naive_Bayes', accuracy_score(y_test, NB_yhat), jaccard_score(y_test, NB_yhat,pos_label='1'), f1_score(y_test, NB_yhat, average='weighted')],

['XGB Classifier', accuracy_score(y_test, XGB_pred), jaccard_score(y_test, XGB_pred,pos_label='1'), f1_score(y_test, XGB_pred, average='weighted')]]

print (tabulate(data, headers=["Model", "Accuracy", "Jaccard score", "F1-Score"]))
prediction_submit = RanFor_model.predict(X_submit)



testdata = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': prediction_submit})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")