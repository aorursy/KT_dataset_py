import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import seaborn as sns

sns.set_style("whitegrid")



from collections import Counter



import warnings

warnings.filterwarnings('ignore')



# plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.offline as pyOff



import colorlover as cl



# widgets

import ipywidgets as widgets

from ipywidgets import interact, interactive



#importing cufflinks in offline mode

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_df['PassengerId']
@interact



def preview_dataframe(preview = ['Head', 'Description', 'Shape', 'Unique', 'Types', 'Columns']):

    

    preview_return = {

                      'Head':train_df.head(10), 

                      'Description':train_df.describe(), 

                      'Shape':train_df.shape, 

                      'Unique':train_df.nunique(axis = 0), 

                      'Types':train_df.dtypes,

                      'Columns':train_df.columns

                      }

    to_display = preview_return[preview]

    return to_display
train_df.info()
def bar_plot(variable):

    """

        input: variable example: 'Survived'

        output: bar plot & value count

    """

    # get variable

    var = train_df[variable]

    # count number of categorical variable

    varValue = var.value_counts()

    

    # visualization

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show()

    print("{}:\n{}".format(variable, varValue))
category1 = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Cabin', 'Name', 'Ticket', 'SibSp', 'Parch']

for c in category1:

    bar_plot(c)
category2 = ['Cabin', 'Name', 'Ticket']

for c in category2:

    print('{}\n'.format(train_df[c].value_counts()))
def hist_plot(variable):

    # visualization

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel('Frequency')

    plt.title('{} distribution with histogram'.format(variable))

    plt.show()
numericVar = ['PassengerId', 'Age', 'Fare']

for n in numericVar:

    hist_plot(n)
cnt_srs = train_df['Age'].value_counts()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        #color = np.random.randn(500), #set color equal to a variable

        color = cnt_srs.values,

        line=dict(width=0.8)

    ),

)

layout = go.Layout(

    title='Age distribution'

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

pyOff.iplot(fig, filename="age")
# Pclass - Survived

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Sex - Survived

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# SibSp - Survived

train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
# Parch - Survived

train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
def detect_outlier(df, features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier Step

        outlier_step = IQR * 1.5

        # Outlier and their indices

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step )].index

        # Store indices

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v >2)

    

    return(multiple_outliers)

    

    
train_df.loc[detect_outlier(train_df, ['Age', 'SibSp', 'Parch', 'Fare'])]
# Drop outliers

tran_df = train_df.drop(detect_outlier(train_df, ['Age', 'SibSp', 'Parch', 'Fare']), axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column = 'Fare', by = 'Embarked')

plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna('C')

train_df[train_df['Embarked'].isnull()]
train_df[train_df['Fare'].isnull()]
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))

train_df[train_df['Fare'].isnull()]
list1 = ['SibSp', 'Parch', 'Age', 'Fare', 'Survived']

sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f')
train_df.corr().iplot(kind='heatmap',colorscale="Blues",title="Feature Correlation Matrix")
def go_hist_trace(data, name, colour='rgba(171, 50, 96, 0.6)'):

    trace = go.Histogram(x=data,

                    opacity=0.75,

                    name = name,

                    xbins=dict(

                        start=0,

                        end=80,

                        size=4

                    ), 

                    autobinx = False,

                    marker=dict(color= colour))

    return trace
y = sns.factorplot(x= "Pclass", y="Survived",data = train_df, kind='bar', size = 6)

y.set_ylabels("Survived Probability")

plt.show

# import graph objects as "go"

age_survived = train_df[train_df.Survived == 1]['Age']

age_not_survived = train_df[train_df.Survived == 0]['Age']



trace1 = go_hist_trace(data=age_survived,name='Survived',colour='rgba(171, 50, 96, 0.6)')

trace2 = go_hist_trace(data=age_not_survived,name='Not survived',colour='rgba(12, 50, 196, 0.6)')



data = [trace1, trace2]



updatemenus = list([

            dict(buttons=list([

            dict(label = 'Both', method = 'update',

                args = [{'visible': [True, True, True]},{'title': 'Survived and Not Survived'}]),

            dict(label = 'Not Survived', method = 'update',

                 args = [{'visible': [False, True, False]},{'title': 'Not Survived',}]),

            dict(label = 'Survived', method = 'update',

                 args = [{'visible': [True, False, False]},{'title': 'Survived',}])

        ]),direction = 'down', showactive = True )])



layout = dict(title = 'Survive by Age',yaxis=dict(title='Count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),

               xaxis= dict(title= 'Age',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=50,r=50),paper_bgcolor='rgb(105,105,105)',

               plot_bgcolor='rgb(105,105,105)',barmode='stack',font= {'color': '#FFFFFF'},updatemenus=updatemenus,showlegend=True)



fig = dict(data = data, layout=layout)

pyOff.iplot(fig, filename='relayout_option_dropdown')
# create trace 1 that is 3d scatter, it is not good here 3d but i wanted to add

trace1 = go.Scatter3d(

    x=train_df.Survived,

    y=train_df.Pclass,

    z=train_df.Age,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")

g.map(plt.hist, "Age", bins=20)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked")

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","Parch","SibSp","Pclass"]].corr(), annot = True)
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    age_med = train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else:

        train_df["Age"].iloc[i] = age_med
train_df[train_df["Age"].isnull()]
train_df["Name"].head(5)
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)
sns.countplot(x= "Title", data = train_df)

plt.xticks(rotation= 90)

plt.show()
#category

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

train_df["Title"].head(10)
sns.countplot(x="Title", data = train_df)

plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True)

train_df.head()
train_df = pd.get_dummies(train_df,columns=["Title"])

train_df.head()
train_df["Embarked"].head()
sns.countplot(x = "Embarked", data = train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()
train_df["Ticket"].head(10)
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df["Ticket"].head(10)
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

train_df.head(10)
sns.countplot(x = "Pclass", data = train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
test.head()
train = train_df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) 

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)