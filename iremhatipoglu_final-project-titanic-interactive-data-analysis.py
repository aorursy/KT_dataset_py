import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt #matplotlib and seaborn are for the graphics that we are going to use

import seaborn as sns

import xgboost as xgb #for modelling

%matplotlib inline

color = sns.color_palette()

import warnings

warnings.filterwarnings('ignore')



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.express as px

import colorlover as cl



pd.options.mode.chained_assignment = None



# widgets

import ipywidgets as widgets

from ipywidgets import interact, interactive



#importing cufflinks in offline mode

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



# Input data files are available in the "../input/" directory.

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#We added useful libraries
train = pd.read_csv("/kaggle/input/titanic/train.csv") #We loaded dataset

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test['PassengerId']

full = pd.concat([train, test], keys=['train','test'])

@interact



def preview_dataframe(preview = ['Head', 'Description', 'Shape', 'Unique', 'Types', 'Columns']):

    

    preview1 = {

                      'Head':test.head(10), #I used test data to see names. When train data was used, it was shown in binary format. 

                      'Description':test.describe(), 

                      'Shape':test.shape, 

                      'Unique':test.nunique(axis = 0), 

                      'Types':test.dtypes,

                      'Columns':test.columns

                      }

    show = preview1[preview]

    return show
train.count() #We used count method for to learn how many people were on the Titanic
train.describe() #Descriptive statistics table of the data set
train.info() #We used for to see data types of columns
#this graphic shows how many men and women on the ship

gender = train['Sex'].value_counts()

trace = go.Bar(

    x=gender.index,

    y=gender.values,

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = gender.values,

        colorscale='Viridis',

        showscale=False

    ),

)

layout = go.Layout(

    title='Gender Distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="sex")
sns.countplot(x='Survived', hue='Sex', data=train) #Gender based survivors
#Number of passengers according to class

train['Pclass'].value_counts()
#this graphic shows how many people have traveled in which class

pclassnum = train['Pclass'].value_counts()

trace = go.Bar(

    y=pclassnum.index,

    x=pclassnum.values,

    orientation = 'h',

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = pclassnum.values,

        colorscale='PuOr',

        showscale=False

    ),

)

layout = go.Layout(

    title='Passenger Class Distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="pclass")
sns.countplot(x='Pclass', hue='Sex', data=train) 
train[train["Name"].str.contains("Brown")] #we seached names contains "Brown" as her lastname.
sns.countplot(x='Survived', hue='Pclass', data=train) #surviving numbers for class of travel
#We diversified age groups by defining a function

def age_dis(x):

    if x>=0 and x <12: #we accepted that the age under 12 years old are child

        return 'Child'

    elif x>=12 and x<=20:

        return 'Young'

    else:

        return 'Adult' #we accepted that the age above 20 years old are adult
train['Age'].apply(age_dis).value_counts() #age based numbers
agepie = train['Age'].apply(age_dis).value_counts()

labels = (np.array(agepie.index))

sizes = (np.array((agepie / agepie.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Age Distribution with Pie Chart'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="agedis")
agebar = train['Age'].value_counts()

trace = go.Bar(

    x=agebar.index,

    y=agebar.values,

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = agebar.values,

        colorscale='Viridis',

        showscale=True

    ),

)

layout = go.Layout(

    title='Age Distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="age")
#Women Survival Status

women = train[train['Sex']=='female']

w_survived = women[women['Survived']==1].Age.dropna()

w_not_survived = women[women['Survived']==0].Age.dropna()

trace1 = go.Histogram(x=w_survived,name='Survived',opacity=0.75)

trace2 = go.Histogram(x=w_not_survived,name='Not survived',opacity=0.75)



data = [trace1, trace2]



# Add dropdown

updatemenus = list ([

            dict(buttons=list([

            dict(label = 'Both', 

                 method = 'update',

                 args = [{'visible': [True, True]},{'title': 'Survival Status of Women by Age'}]),

            dict(label = 'Not Survived', 

                 method = 'update',

                 args = [{'visible': [False, True]},{'title': 'Not Survived',}]),

            dict(label = 'Survived', 

                 method = 'update',

                 args = [{'visible': [True, False]},{'title': 'Survived',}])

        ]), direction="down",

            pad={"r": 10, "t": 10},

                 showactive = True )])



layout = dict(title = 'Survival Status of Women',

            yaxis=dict(title='Count',linecolor='rgba(255,255,255)',showgrid=True,gridcolor='rgba(255,255,255)'),

            xaxis= dict(title= 'Age',linecolor='rgba(255,255,255)',showgrid=True,gridcolor='rgba(255,255,255)'),margin=go.Margin(l=50,r=50),barmode='overlay',

            font= {'color': '#000000'},updatemenus=updatemenus,showlegend=True)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='femaleAgeSurvival')
#Men Survival Status

men = train[train['Sex']=='male']

m_survived = men[men['Survived']==1].Age.dropna()

m_not_survived = men[men['Survived']==0].Age.dropna()

trace1 = go.Histogram(x=m_survived,name='Survived',opacity=0.75)

trace2 = go.Histogram(x=m_not_survived,name='Not survived',opacity=0.75)



data = [trace1, trace2]



# Add dropdown

updatemenus = list([

            dict(buttons=list([

            dict(label = 'Both', 

                 method = 'update',

                 args = [{'visible': [True, True]},{'title': 'Survival Status of Men by Age'}]),

            dict(label = 'Not Survived', 

                 method = 'update',

                 args = [{'visible': [False, True]},{'title': 'Not Survived',}]),

            dict(label = 'Survived', 

                 method = 'update',

                 args = [{'visible': [True, False]},{'title': 'Survived',}])

        ]), direction="down",

            pad={"r": 10, "t": 10},

                 showactive = True )])



layout = dict(title = 'Survival Status of Men',

            yaxis=dict(title='Count',linecolor='rgba(255,255,255)',showgrid=True,gridcolor='rgba(255,255,255)'),

            xaxis= dict(title= 'Age',linecolor='rgba(255,255,255)',showgrid=True,gridcolor='rgba(255,255,255)'),margin=go.Margin(l=50,r=50),barmode='overlay',

            font= {'color': '#000000'},updatemenus=updatemenus,showlegend=True)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='maleAgeSurvival')
ys = [train.Age.values]

xs = [train.Pclass.values]

names = ["1", "2", "3"]



trace = []



for i in range(1):

    trace.append ( 

        go.Box(

            y=ys[i],

            x=xs[i],

            name=names[i],

            marker = dict(

            )

        )

    )



layout = go.Layout(

    title='Age Distribution by Passenger Class'

)

#data = [trace0, trace1]

fig = go.Figure(data=trace, layout=layout)

py.iplot(fig, filename="agepclass")
#We are going to spot some more features, that contain missing values (NaN = not a number)



total = train.isnull().sum().sort_values(ascending=False)

percent_1 = train.isnull().sum()/train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train["Embarked"] = train["Embarked"].fillna('S') #filled with S
sns.countplot(x='Survived', hue='Embarked', data=train) #Surviving rates based on embarking spots
def add_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        return int(train[train["Pclass"] == Pclass]["Age"].mean()) #We obtained the average age with .mean () function

    else:

        return Age
train["Age"] = train[["Age", "Pclass"]].apply(add_age,axis=1) #we call the function
train.drop("Cabin",inplace=True,axis=1) #we removed Cabin with .drop() function
train.isnull().sum() #we removed rows with null values
name = train['Name']

train['Title'] = [i.split(".")[0].split(",")[-1].strip() for i in name] #we split and create a new feature
train['Title'].head(10) #to see first 10 rows of our feature
#This is Title distribution with pie chart

titlepie = train['Title'].value_counts()

labels = (np.array(titlepie.index))

sizes = (np.array((titlepie / titlepie.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Title Distribution with Pie Chart'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="titledis")
data = [train]

titles = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with Rare

    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev','Dr', 'Major',\

                                            'Lady', 'Sir', 'Col', 'Capt','the Countess','Jonkheer'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'], axis=1)

#This is updated Title graphic

title = train['Title'].value_counts()

trace = go.Bar(

    x=title.index,

    y=title.values,

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = title.values,

        colorscale='Rainbow',

        showscale=False

    ),

)

layout = go.Layout(

    title='Updated Title Graphic'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="title")



#INFORMATION

# 0 : Mr

# 1 : Mrs

# 2 : Miss

# 3 : Master

# 4 : Rare
train = pd.get_dummies(train,columns=["Title"])

train.head()
data = [train]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train]



for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)
train["Sex"] = train["Sex"].astype("category")

train = pd.get_dummies(train, columns=["Sex"])

train.head()
train['Ticket'].describe()
train= train.drop(["Ticket", "PassengerId"], axis=1) # I also dropped Passenger Id  too cause it's unnecessary.
ports = {"S": 0, "C": 1, "Q": 2}

data = [train]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
train = pd.get_dummies(train, columns=["Embarked"])

train.head()
train['Pclass'] = train['Pclass'].astype("category")

train = pd.get_dummies(train, columns= ['Pclass'])

train.head()
#importing necessary libraries

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from scipy.stats.stats import pearsonr

from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler
X = train.drop("Survived",axis=1) #x will contain all the features and y will contain the target variable

y = train["Survived"]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))

# Making a list of accuracies

accuracies = []
rdmf = RandomForestClassifier(n_estimators=20, criterion='entropy')

rdmf.fit(X_train, y_train)
#writing the accuracy score

rdmf_score = rdmf.score(X_test, y_test)

rdmf_score_tr = rdmf.score(X_train, y_train)

accuracies.append(rdmf_score)

print(rdmf_score)

print(rdmf_score_tr)
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
#writing the accuracy score

lr_score = classifier.score(X_test, y_test)

accuracies.append(lr_score)

print(lr_score)
knn = KNeighborsClassifier(p=2, n_neighbors=10)

knn.fit(X_train, y_train)
#writing the accuracy score

knn_score = knn.score(X_test, y_test)

accuracies.append(knn_score)

print(knn_score)
svm = SVC(kernel='linear')

svm.fit(X_train, y_train)
#writing the accuracy score

svm_score = svm.score(X_test, y_test)

accuracies.append(svm_score)

print(svm_score)
k_svm = SVC(kernel='rbf')

k_svm.fit(X_train, y_train)
#writing the accuracy score

k_svm_score = k_svm.score(X_test, y_test)

accuracies.append(k_svm_score)

print(k_svm_score)
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
#writing the accuracy score

xgb_score = xgb.score(X_test, y_test)

accuracies.append(xgb_score)

print(xgb_score)
accuracy_labels = ['Random Forest', 'Logistic Regression', 'KNN', 'LSVM', 'Kernel SVM', 'Xgboost']
#this graphic shows accuracy scores



trace = go.Bar(

    y=accuracy_labels,

    x=accuracies,

    orientation = 'h',

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = pclassnum.values,

        colorscale='RdGy',

        showscale=False

    ),

)

layout = go.Layout(

    title='Accuracy Scores'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="accuracy")
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

#confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
#prediction and submission

test_survived = pd.Series(classifier.predict(X_test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic_output.csv", index = False) #output