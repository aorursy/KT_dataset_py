import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

# Scalers

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle



# Models

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.linear_model import Perceptron

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier



# Cross-validation

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.model_selection import cross_validate



# GridSearchCV

from sklearn.model_selection import GridSearchCV



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

data_dir = '../input/titanic/'

os.listdir(data_dir)
data = pd.read_csv(data_dir+'train.csv')

test_df = pd.read_csv(data_dir+'test.csv')

df = data.append(test_df) # The entire data: train + test.
df.info()
df.describe()
df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df['Embarked'].value_counts()

M = df[(df['Embarked'] == 'S')]

B = df[(df['Embarked'] == 'C')]

K  = df[(df['Embarked'] == 'Q')]



trace = go.Bar(x = (len(M), len(B),len(K)), y = ['S','C','Q'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['red','green','blue'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Count of target variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
labels = data['Sex'].value_counts()[:10].index

values = data['Embarked'].value_counts()[:10].values

colors=['#2678bf','#98adbf']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.show()
fig = px.scatter(data, x="Age", y="Fare", color="Survived", facet_col="Survived",

           color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")

fig.show()
fig = px.scatter(data, x="Age", y="Fare", color="Sex", facet_col="Sex",

           color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")

fig.show()
data_to_plot = data.dropna()

fig = px.violin(data_to_plot, y="Fare", x="Embarked", color="Embarked", box=True, points="all")

fig.show()
#correlation map

f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, linewidth=".5", fmt=".2f", ax = ax)

plt.title("Correlation Map",fontsize=20)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Embarked'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Class Count')

ax[0].set_ylabel('Count')

sns.countplot('Embarked',data=df,ax=ax[1],order=df['Embarked'].value_counts().index)

ax[1].set_title('Count of Target')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Embarked","Age", hue="Embarked", data=df,ax=ax[0])

ax[0].set_title('Age and Target')

sns.violinplot("Embarked","Fare", hue="Embarked", data=df,ax=ax[1])

ax[1].set_title('Fare  and Target')

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,10))

df[df['Sex']=='male'].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Sex:male')



df[df['Sex']=='female'].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Sex:female')



plt.show()
pd.crosstab(data.Age,data.Survived).plot(kind="bar",figsize=(20,6))

plt.title('Survived Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(data.Pclass,data.Survived).plot(kind="bar",figsize=(15,6),color=['#DAA7A6','#FF5933' ])

plt.title('Pclass Distribution')

plt.xlabel('Pclass')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()

pd.crosstab(data.Embarked,data.Survived).plot(kind="bar",figsize=(15,6),color=['#11E5AA','#BB1190' ])

plt.title('Embarked Frequenct w.r.t Survived')

plt.xlabel('Embarked')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
data['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace = True)



# Making Bins

df['FareBin'] = pd.qcut(df['Fare'], 5)



label = LabelEncoder()

df['FareBin_Code'] = label.fit_transform(df['FareBin'])



data['FareBin_Code'] = df['FareBin_Code'][:891]

test_df['FareBin_Code'] = df['FareBin_Code'][891:]



data.drop(['Fare'], 1, inplace=True)

test_df.drop(['Fare'], 1, inplace=True)
df['Age'].fillna(df['Fare'].median(), inplace = True)



df['AgeBin'] = pd.qcut(df['Age'], 4)



label = LabelEncoder()

df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])



data['AgeBin_Code'] = df['AgeBin_Code'][:891]

test_df['AgeBin_Code'] = df['AgeBin_Code'][891:]



data.drop(['Age'], 1, inplace=True)

test_df.drop(['Age'], 1, inplace=True)
data['fam_count'] = data['SibSp']+data['Parch']

test_df['fam_count'] = test_df['SibSp']+test_df['Parch']



size = {

    0:'alone',

    1:'small',

    2:'small',

    3:'small',

    4:'large',

    5:'large',

    6:'large',

    7:'large',

    10:'large'

}



data['family_size'] = data['fam_count'].map(size)

test_df['family_size'] = test_df['fam_count'].map(size)
def create_dummies(df, column_name):

    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df
data = create_dummies(data,'Embarked')

data = create_dummies(data,'Pclass')

data = create_dummies(data,'family_size')
test_df = create_dummies(test_df,'Embarked')

test_df = create_dummies(test_df,'Pclass')

test_df = create_dummies(test_df,'family_size')
df_test = test_df.copy()
data.head()
data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis = 1, inplace = True)

test_df.drop(['Name','PassengerId', 'Ticket', 'Cabin'], axis = 1, inplace = True)
def get_X_and_y(dataset, target_name):

    X=dataset.drop(target_name, axis=1)

    y=dataset[target_name]

    sc = StandardScaler()

    X=sc.fit_transform(X)    

    return X, y
X,y = get_X_and_y(data,target_name='Survived')
model = RandomForestClassifier(n_estimators = 100, max_depth = 5,random_state = 1)

model.fit(X,y)

predictions = model.predict(test_df)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")