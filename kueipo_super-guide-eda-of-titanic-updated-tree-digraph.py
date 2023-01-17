import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

%matplotlib inline

import plotly # visualization

import plotly.tools as tls # visualization

import plotly.plotly as py # visualization

from plotly.graph_objs import Scatter, Figure, Layout # visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot # visualization

import plotly.figure_factory as ff # visualization

import plotly.graph_objs as go # visualization

init_notebook_mode(connected=True) # visualization

import missingno as msno # visualization



from sklearn.preprocessing import LabelEncoder 

from sklearn.cross_validation import KFold

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.metrics import confusion_matrix # Metric 

# Ensemble

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier 

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

import xgboost as xgb # Gradeint Boosting

from xgboost import XGBClassifier # Gradeint Boosting

import lightgbm as lgb # Gradeint Boosting

import catboost as cat  # Gradeint Boosting



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



combine = pd.concat([train.drop('Survived',1),test])
train.head()
class1_s = len(train[(train['Pclass']==1) & (train['Survived']==1)])

class1_d = len(train[(train['Pclass']==1) & (train['Survived']==0)])

class2_s = len(train[(train['Pclass']==2) & (train['Survived']==1)])

class2_d = len(train[(train['Pclass']==2) & (train['Survived']==0)])

class3_s = len(train[(train['Pclass']==3) & (train['Survived']==1)])

class3_d = len(train[(train['Pclass']==3) & (train['Survived']==0)])



trace1 = go.Bar(

    x=['First Class', 'Bussiness Class', 'Economy Class'],

    y=[class1_s , class2_s , class2_s],

    name='Survive')



trace2 = go.Bar(

    x=['First Class', 'Bussiness Class', 'Economy Class'],

    y=[class1_d , class2_d , class2_d],

    name='Be with God',

    marker=dict(

        color='rgb(58,200,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)))

data = [trace1, trace2]

layout = go.Layout(

    barmode='group',

    yaxis = dict(zeroline = False, title='Counts'))

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='grouped-bar')
fig = {

  "data": [

    {

      "values": [class1_s ,class2_s ,class3_s],

      "labels": ['Noble Class', 'Bussiness Class', 'Poor Class'],

      "domain": {"x": [0, .48]},

      "name": "Survived",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      'marker': {'colors': ['rgb(58,200,225)','rgb(129, 180, 179)']},

      "type": "pie"

    },     

    {

      "values": [class1_d ,class2_d ,class3_d],

      "labels": ['Noble Class', 'Bussiness Class', 'Poor Class'],

      "text":"Be with God",

      "textposition":"inside",

      "domain": {"x": [.52, 1]},

      "name": "In Heaven",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Titanic Survival/Dead Ratio",

        "annotations": [

            {

                "font": {"size": 20},

                "showarrow": False,

                "text": "Alive",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {"size": 20},

                "showarrow": False,

                "text": "Dead",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}

plotly.offline.iplot(fig, filename='donut')
c1_y = len(train[(train['Sex']=='female')&(train['Age']<=18)& (train['Survived']==1)])

c1_a = len(train[(train['Sex']=='female')&((train['Age']>=18)&(train['Age']<45))& (train['Survived']==1)])

c1_e = len(train[(train['Sex']=='female')&(train['Age']>=45)& (train['Survived']==1)])



c2_y = len(train[(train['Sex']=='male')&(train['Age']<=18)& (train['Survived']==1)])

c2_a = len(train[(train['Sex']=='male')&((train['Age']>=18)&(train['Age']<45))& (train['Survived']==1)])

c2_e = len(train[(train['Sex']=='male')&(train['Age']>=45)& (train['Survived']==1)])





trace1 = go.Bar(

    x=['Young', 'Adult', 'Elder'],

    y=[c1_y, c1_a, c1_e],

    name='Female',

    marker=dict(

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        )

)

trace2 = go.Bar(

    x=['Young', 'Adult', 'Elder'],

    y=[c2_y, c2_a, c2_e],

    name='Male',

    marker=dict(

        color='rgb(58,200,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5),

        )

)



data = [trace1, trace2]

layout = go.Layout(

    title = 'Titanic Suvival Gender Ratio',

    barmode='stack',

    yaxis = dict(title='Counts')

)



fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='stacked-bar')



trace0 = go.Scatter(

    y = train[(train['Survived']==1)]['Fare'].values,

    x = train[(train['Survived']==1)]['Age'].values,

    name = 'Survived',

    mode = 'markers',

    marker = dict(

        size = 8,

        color = 'rgb(58,200,225)',

        line = dict(

            width = 2,

            color = 'rgb(0, 0, 0)'

        )

    )

)



trace1 = go.Scatter(

    y = train[(train['Survived']==0)]['Fare'].values,

    x = train[(train['Survived']==0)]['Age'].values,

    name = 'Dead',

    mode = 'markers',

    marker = dict(

        size = 6,

        color = 'rgba(255, 182, 193, .9)',

        line = dict(

            width = 2,

        )

    )

)



data = [trace0, trace1]



layout = dict(title = 'Styled Scatter of Fare Vs Age',

              yaxis = dict(zeroline = False, title='Fare'),

              xaxis = dict(zeroline = False, title='Age')

             )



fig = dict(data=data, layout=layout)

plotly.offline.iplot(fig, filename='styled-scatter')

train[train['Pclass']==3][['Fare']].max()
print('The survival Rate when Fare over 70$ is :',round(train[(train['Fare']>=70)&(train['Survived']==1)].shape[0]/train[(train['Fare']>=70)].shape[0],5)*100,'%')
# Create a new feature Title

combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

combine.groupby(['Sex'])['Title'].value_counts()
combine['Title'] = combine['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 

                                             'Jonkheer', 'Dona'], 'Rare')



combine['Title'] = combine['Title'].replace('Mlle', 'Miss')

combine['Title'] = combine['Title'].replace('Ms', 'Miss')

combine['Title'] = combine['Title'].replace('Mme', 'Mrs')

    
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                         'Don', 'Dr', 'Major', 'Rev', 'Sir', 

                                         'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
# Gives the length of the name, exclude Title

combine['Name_length'] = combine['Name'].apply(len) - combine['Title'].apply(len)
trace1 = go.Bar(

    x=['Master', 'Miss', 'Mr', 'Mrs', 'Rare'],

    y=train[train['Survived']==1][['Title', 'Survived']].groupby(['Title']).size(),

    name='Survived',

    marker = dict(

        color = 'rgba(255, 182, 193, .9)',

    )

)

trace2 = go.Bar(

    x=['Master', 'Miss', 'Mr', 'Mrs', 'Rare'],

    y=train[train['Survived']==0][['Title', 'Survived']].groupby(['Title']).size(),

    name='Dead',

    marker = dict(

        color = 'rgb(255, 0, 0)',

    )

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    yaxis = dict(zeroline = False, title='Count')

)

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='stacked-bar')
combine['FirstName'] = combine['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]
combine['LuckyElizabeth'] = np.where((combine['FirstName']=='Elizabeth'),1,0)

combine['LuckyAnna'] = np.where((combine['FirstName']=='Anna'),1,0)

combine['LuckyMary'] = np.where((combine['FirstName']=='Mary'),1,0)
combine['Family'] = combine["Parch"] + combine["SibSp"]

train['Family'] = train["Parch"] + train["SibSp"]
trace1 = go.Bar(

    x=train[train['Survived']==1][['Family', 'Survived']].groupby(['Family']).size().index,

    y=train[train['Survived']==1][['Family', 'Survived']].groupby(['Family']).size(),

    name='Survived',

    marker = dict(

        color = 'rgba(255, 182, 193, .9)',

    )

)

trace2 = go.Bar(

    x=train[train['Survived']==0][['Family', 'Survived']].groupby(['Family']).size().index,

    y=train[train['Survived']==0][['Family', 'Survived']].groupby(['Family']).size(),

    name='Dead',

    marker = dict(

        color = 'rgb(255, 0, 0)',

    )

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    yaxis = dict(zeroline = False, title='Count'),

    xaxis = dict(zeroline = False, title='Family Size',dtick=1)

)

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='stacked-bar')
for i in np.sort(train['Family'].unique()):

    print('Has died, Male (Family Size: {}) proportion: {}'.format(i,

      round(train[(train['Sex']=='male')&(train['Survived']==0)&(train['Family']==i)].shape[0]/\

          train[(train['Survived']==0)&(train['Family']==i)].shape[0]*100,2)),'%')
combine['Alone'] = 0

combine.ix[combine['Family'] == 0, 'Alone'] = 1

combine.ix[combine.Family==0,'Fsize'] = 'Single'

combine.ix[(combine.Family>0)&(combine.Family<5),'Fsize'] = 'small'

combine.ix[(combine.Family>4),'Fsize'] = 'big'
x = combine[(combine['Alone']==1)][['Age']].round().dropna().groupby(['Age']).size() 

trace0 = go.Scatter(

    x = x.index,

    y = x,

    mode = 'lines',

    name = 'lines',

    marker = dict(

        color = 'rgba(255, 182, 193, .9)',

    )

)

data = [trace0]

plotly.offline.iplot(data, filename='line-mode')
combine.isnull().sum()
# Using msno library to see the NaN distribution

missingValueColumns = combine.columns[combine.isnull().any()].tolist()

msno.matrix(combine[missingValueColumns],width_ratios=(10,1),\

            figsize=(20,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)

plt.show()
combine[(combine['Embarked']=='S')&(combine['Sex']=='male')&(combine['Pclass']==3)&

        (combine['Age']>60)&(combine['Fsize']=='Single')][['Fare']].describe()
combine['Fare'].fillna(7.775, inplace=True)
# Remember that must relax the condition to find the values

combine[(combine['Fare']>70)&(combine['Sex']=='female')&(combine['Pclass']==1)].groupby(['Embarked']).size()
combine[combine['Embarked'].isnull()][['Name']]
combine[combine['Embarked']=='C'][['Name']].head(10)

# Yes, almost French name
c = combine[(combine['Embarked']=='C')&(combine['Pclass']==1)]['Fare']

q = combine[(combine['Embarked']=='Q')&(combine['Pclass']==1)]['Fare']

s = combine[(combine['Embarked']=='S')&(combine['Pclass']==1)]['Fare']



trace0 = go.Box(

    y=c,

    name = 'C',

    marker = dict(

        color = 'rgb(214, 12, 140)',

    )

)

trace1 = go.Box(

    y=q,

    name = 'Q',

    marker = dict(

        color = 'rgb(0, 128, 128)',

    )

)

trace2 = go.Box(

    y=s,

    name = 'S',

    marker = dict(

        color = 'rgb(107,174,214)',

    )

)

data = [trace0, trace1,trace2 ]

plotly.offline.iplot(data)
combine['Embarked'].fillna('C', inplace=True)
# Fill NaN for First Class single male/female

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==1)&(combine['Sex']=='female'),'Age'] = 35

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==1)&(combine['Sex']=='male'),'Age'] = 42



# Fill NaN for Second Class single male/female

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==2)&(combine['Sex']=='female'),'Age'] = 30

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==2)&(combine['Sex']=='male'),'Age'] = 30



# Fill NaN for Third Class single male/female

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==3)&(combine['Sex']=='female'),'Age'] = 22

combine.ix[(combine['Age'].isnull())&(combine['Pclass']==3)&(combine['Sex']=='male'),'Age'] = 26



if combine['Age'].isnull().sum() == 0:

    print('Done, Age Missing Value accomplished!! 乁( ◔ ౪◔)「')
# Let's see the Cabin Number

combine['Cabin'].unique()
combine[combine['Cabin'].isnull()][['Pclass']].groupby(['Pclass']).size()
combine.ix[(combine.Pclass==1)&(combine.Cabin.isnull()),'Cabin'] = np.random.choice(['A','B','C','D'])

combine.ix[(combine.Pclass==2)&(combine.Cabin.isnull()),'Cabin'] = np.random.choice(['D','E','F'])

combine.ix[(combine.Pclass==3)&(combine.Cabin.isnull()),'Cabin'] = np.random.choice(['E','F','G'])



if combine['Cabin'].isnull().sum() == 0:

    print('Beautiful, no more Cabin NaN. ლ(＾ω＾ლ)')
combine['Cabin_Lv'] = combine.Cabin.str.extract('(^.{0,1})', expand=False)

combine['Cabin_Lv'].unique()

# Beautiful no more NaN
combine.columns
# Check categorical column names

combine.select_dtypes(include=['object']).columns
combine.drop(['Name','Ticket','Cabin', 'SibSp', 'Parch','FirstName'], axis = 1, inplace = True)



# encode categorical features

catego_features = ['Sex', 'Embarked', 'Title', 'Fsize', 'Cabin_Lv']

catego_le = LabelEncoder()



for i in catego_features:

    combine[i] = catego_le.fit_transform(combine[i])
corrmat = combine.drop('PassengerId',axis=1).corr()

f, ax = plt.subplots(figsize=(12, 9))

plt.title("Features' Correlation",size=15)

sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8, fmt='.2f');
train_df = combine.iloc[:train.shape[0],:]

train_df['Survived'] = train['Survived']

test_df = combine.iloc[train.shape[0]: ,:]
train_df.head()
X = train_df.drop(['PassengerId','Survived','Alone'], axis=1)

y = train_df.Survived.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
matplotlib.style.use('fivethirtyeight')

matplotlib.rcParams['figure.figsize'] = (12,6)

model = DecisionTreeClassifier(max_depth=6 ,random_state=87)

model.fit(X, y)

feat_names = X.columns.values

## plot the importances ##

importances = model.feature_importances_



indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))

plt.title("Feature importances by DecisionTreeClassifier")

plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
from sklearn.tree import export_graphviz

import graphviz

treegraph = export_graphviz(model, out_file=None, 

                         feature_names=X.columns,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(treegraph)  

graph
model = RandomForestClassifier(n_estimators = 100 ,max_depth=8, max_features=None,

                               min_samples_split = 4,min_samples_leaf=2,random_state=87)

model.fit(X, y)

feat_names = X.columns.values

## plot the importances ##

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



plt.figure(figsize=(12,6))

plt.title("Feature importances by Random Forest")

plt.bar(range(len(indices)), importances[indices], color='lightblue', yerr=std[indices], align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
model = XGBClassifier(eta = 0.05, max_depth = 7, subsample = 0.8, colsample_bytree= 0.4,

                     num_iterations= 7000, max_leaves=4)

model.fit(X, y)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))

plt.title("Feature importances by XGB") # Thanks Oscar Takeshita's kindly remind

plt.bar(range(len(indices)), importances[indices], color='lightblue', align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
xgb.to_graphviz(model, fmap='', rankdir='UT', num_trees=6,

                yes_color='#0000FF', no_color='#FF0000')
lgb_params = {}

lgb_params['objective'] = 'binary'

lgb_params['metric'] = 'auc'

lgb_params['sub_feature'] = 0.80 

lgb_params['max_depth'] = 8

lgb_params['feature_fraction'] = 0.7

lgb_params['bagging_fraction'] = 0.7

lgb_params['bagging_freq'] = 10

lgb_params['learning_rate'] = 0.01

lgb_params['num_iterations'] = 5000



lgb_train = lgb.Dataset(X, y)

lightgbm = lgb.train(lgb_params, lgb_train)



lgb.plot_importance(lightgbm)

plt.title("Feature importances by LightGBM")

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.show()
lgb.create_tree_digraph(lightgbm)