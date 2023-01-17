import psycopg2

import sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from itertools import cycle

from sklearn import svm, datasets

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp



# Python libraries

# Classic,data manipulation and linear algebra

import pandas as pd

import numpy as np



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import squarify



# Data processing, metrics and modeling

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score

from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgbm



# Stats

import scipy.stats as ss

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform



# Time

from contextlib import contextmanager

@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 
train = pd.read_csv('C:\\Users\\guillesantillan\\Desktop\\Train.csv', index_col='PassengerId')

test = pd.read_csv('C:\\Users\\guillesantillan\\Desktop\\Test.csv', index_col='PassengerId')
train.head(5)
# Creating variable Title

train['Title'] = train['Name']

# Cleaning name and extracting Title

for name_string in train['Name']:

    train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=True)
# Creating variable Title

test['Title'] = test['Name']

# Cleaning name and extracting Title

for name_string in test['Name']:

    test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=True)
# Replacing rare titles 

mapping = {'Mlle': 1,'Ms': 1,'Miss': 1,'Mr': 2,'Mme': 1,'Mrs': 1,'Major': 4,'Col': 4,'Dr' : 4,'Rev' : 4,'Capt': 4,'Other': 3,

           'Jonkheer': 5,'Sir': 5,'Lady': 500,'Don': 5,'Countess': 5,'Dona': 500,'Royal': 5,'Master': 6}

train.replace({'Title': mapping}, inplace=True)

titles = [1,2,3,4,5,6]
# Replacing rare titles 

mapping = {'Mlle': 1,'Ms': 1,'Miss': 1,'Mr': 2,'Mme': 1,'Mrs': 1,'Major': 4,'Col': 4,'Dr' : 4,'Rev' : 4,'Capt': 4,'Other': 3,

           'Jonkheer': 5,'Sir': 5,'Lady': 500,'Don': 5,'Countess': 5,'Dona': 500,'Royal': 5,'Master': 6}

test.replace({'Title': mapping}, inplace=True)

titles = [1,2,3,4,5,6]
train['Sexo*Edad'] = round(train.Sex*train.Age, 2)

train['Sexo*Embarked'] = round(train.Sex*train.Embarked, 2)

train['Edad*Embarked'] = round(train.Age*train.Embarked, 2)

train['Fare*Embarked'] = round(train.Age*train.Fare, 2)

train['Age*Parch'] = round(train.Age*train.Parch, 2)

train['SibSp*Parch'] = round(train.SibSp*train.Parch, 2)

train['Embarked*Parch'] = round(train.Embarked*train.Parch, 2)

train['SibSp*Embarked'] = round(train.SibSp*train.Embarked, 2)

train['SibSp*Age'] = round(train.SibSp*train.Age, 2)

train['Title*Age'] = round(train.Title*train.Age, 2)

train['Title*Sex'] = round(train.Title*train.Sex, 2)

train['Title*Fare'] = round(train.Title*train.Fare, 2)

train['Title*Pclass'] = round(train.Title*train.Pclass, 2)

train['Title*Cabin'] = round(train.Title*train.Cabin, 2)

train['Title*SibSp'] = round(train.Title*train.SibSp, 2)

train['Title*Embarked'] = round(train.Title*train.Embarked, 2)

train['Cabin*Fare'] = round(train.Cabin*train.Fare, 2)



train['Fare*Embarked*Title*Pclass'] = round(train.Age*train.Fare*train.Title*train.Pclass, 2)

train['Cabin*Fare*train*Embarked'] = round(train.Cabin*train.Fare*train.Title*train.Embarked, 2)



train['Edad+Embarked'] = round(train.Age+train.Embarked, 2)

train['Fare+Embarked'] = round(train.Age+train.Fare, 2)

train['Age+Parch'] = round(train.Age+train.Parch, 2)

train['SibSp+Parch'] = round(train.SibSp+train.Parch, 2)

train['SibSp+Age'] = round(train.SibSp+train.Age, 2)

train['Cabin+Fare'] = round(train.Cabin+train.Fare, 2)
test['Sexo*Edad'] = round(test.Sex*test.Age, 2)

test['Sexo*Embarked'] = round(test.Sex*test.Embarked, 2)

test['Edad*Embarked'] = round(test.Age*test.Embarked, 2)

test['Fare*Embarked'] = round(test.Age*test.Fare, 2)

test['Age*Parch'] = round(test.Age*test.Parch, 2)

test['SibSp*Parch'] = round(test.SibSp*test.Parch, 2)

test['Embarked*Parch'] = round(test.Embarked*test.Parch, 2)

test['SibSp*Embarked'] = round(test.SibSp*test.Embarked, 2)

test['SibSp*Age'] = round(test.SibSp*test.Age, 2)

test['Title*Age'] = round(test.Title*test.Age, 2)

test['Title*Sex'] = round(test.Title*test.Sex, 2)

test['Title*Sex'] = round(test.Title*test.Sex, 2)

test['Title*Fare'] = round(test.Title*test.Fare, 2)

test['Title*Pclass'] = round(test.Title*test.Pclass, 2)

test['Title*Cabin'] = round(test.Title*test.Cabin, 2)

test['Title*SibSp'] = round(test.Title*test.SibSp, 2)

test['Title*Embarked'] = round(test.Title*test.Embarked, 2)

test['Cabin*Fare'] = round(test.Cabin*test.Fare, 2)



test['Fare*Embarked*Title*Pclass'] = round(test.Age*test.Fare*test.Title*test.Pclass, 2)

test['Cabin*Fare*train*Embarked'] = round(test.Cabin*test.Fare*test.Title*test.Embarked, 2)



test['Edad+Embarked'] = round(test.Age+test.Embarked, 2)

test['Fare+Embarked'] = round(test.Age+test.Fare, 2)

test['Age+Parch'] = round(test.Age+test.Parch, 2)

test['SibSp+Parch'] = round(test.SibSp+test.Parch, 2)

test['SibSp+Age'] = round(test.SibSp+test.Age, 2)

test['Cabin+Fare'] = round(test.Cabin+test.Fare, 2)
# Plotting age vs sex vs target

g = sns.FacetGrid(train, col="Sex", hue="Survived", palette="Set1")

g.map(sns.distplot, "Age")

g = g.add_legend()

g.fig.suptitle('Age vs Sex vs Survived', fontsize=16)

g.fig.set_size_inches(15,8)
# Plotting title vs age vs sex

g = sns.FacetGrid(train, col="Title", hue="Survived")

g.map(sns.distplot, "Age")

g = g.add_legend()

g.set(ylim=(0, 0.1))

g.fig.suptitle('Title vs Age', fontsize=16)

g.fig.set_size_inches(15,8)
# Defining missing plot to detect all missing values in dataset

def missing_plot(dataset, key) :

    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])

    percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum()))/len(dataset[key])*100, columns = ['Count'])

    percentage_null = percentage_null.round(2)



    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',

            line=dict(color='#000000',width=1.5)))



    layout = dict(title =  "Missing Values (count & %)")



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
# Plotting 

missing_plot(test, 'Age')
y = train.Survived
features = train.columns.values.tolist()

features.remove('Name')

features.remove('Survived')



X = train[features].copy()

X_test = test[features].copy()
# Train_test split

random_state = 0

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.001, random_state = random_state)
y_train.mean()
plt.hist(y_train)

plt.show()
Results = pd.DataFrame({'Model': [],'Accuracy Score': []})
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=666, min_samples_split=20, random_state=0)

model_5 = DecisionTreeClassifier(max_depth=4)

model_6 = RandomForestClassifier(n_estimators=2500, max_depth=4)

model_7 = LogisticRegression()

model_8 = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)
models = {'decission_tree': model_5,

          'random_forest_class': model_6,

          'xgboost': model_8}
for model_name, model in models.items():

    model.fit(X, y)

    y_pred = model.predict(X_valid)

    res = pd.DataFrame({"Model":[model_name], "Accuracy Score": [accuracy_score(y_pred,y_valid)]})

    Results = Results.append(res)
Results
modelo = model_8
plt.figure(figsize=(15,5))

feature_names_x, feature_names_ranks = zip(*sorted(zip(features, modelo.feature_importances_), key=lambda x: x[-1]))

plt.barh(feature_names_x, feature_names_ranks, )

plt.show()
plt.hist(modelo.predict(X_test))

plt.show()
total = 0

count = 0

for i in modelo.predict(X_valid):

    if i == np.asarray(y_valid)[total]:

        count = count + 1

    total = total + 1
round(count/total, 4)
preds_test = modelo.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': preds_test})

output.to_csv('submission.csv', index=False)
output.head()