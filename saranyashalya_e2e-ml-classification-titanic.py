## Importing libraries



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import warnings

import json

import os

from sklearn import preprocessing

from pandas import get_dummies

import xgboost as xgb

import scipy

import seaborn as sns

import matplotlib

import pandas

import numpy

import matplotlib.pylab as pylab
print('matplotlib: {}'.format(matplotlib.__version__))

print('numpy :{}'.format(numpy.__version__))

print('pandas :{}'.format(pandas.__version__))
## for better code readability

sns.set(style ='white', context ='notebook', palette = 'deep')

pylab.rcParams['figure.figsize']=12,8

warnings.filterwarnings('ignore')

matplotlib.style.use('ggplot')

sns.set_style('white')

%matplotlib inline
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
type(df_train), type(df_test)
## scatter plot



g = sns.FacetGrid(df_train, hue ='Survived', col='Pclass', 

                  palette = {1:'seagreen', 0:'gray'})

g = g.map(plt.scatter, 'Fare', 'Age', edgecolor ='w').add_legend()
df_train.plot(kind='scatter', x = 'Age', y='Fare', alpha = 0.5, color='red')
## scatter plot with matplotlib

plt.figure(figsize = (8,4))

plt.scatter(range(df_train.shape[0]), np.sort(df_train['Age'].values))

plt.xlabel("Index")

plt.ylabel("Age")

plt.title("Age trend")

plt.show()
ax = sns.boxplot(x='Pclass', y= 'Age', data = df_train)

ax = sns.stripplot(x='Pclass',y='Age', data = df_train,jitter=True)
## Histogram



df_train.hist(figsize =(15,15))

plt.show()
# Age has normal distribution

df_train['Age'].hist()
df_train.Age.plot(kind='hist', bins=5)
f, ax= plt.subplots(1,2, figsize =(10,5))

df_train[df_train['Survived']==0].Age.plot.hist(ax = ax[0], bins = 20, color='red', edgecolor='black')

ax[0].set_title('Survived=0')

x1 = list(range(0,85,5))

ax[0].set_xticks(x1)



df_train[df_train['Survived']==1].Age.plot.hist(ax = ax[1], bins =20, color='green', edgecolor='black')

ax[1].set_title('Survived=1')

ax[1].set_xticks(x1)

plt.show()
f, ax = plt.subplots(1,2, figsize=(10,5))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax = ax[0], shadow = True)

ax[0].set_title('Survived')

ax[0].set_ylabel("")



sns.countplot('Survived', data = df_train, ax = ax[1])

ax[1].set_title('Survived')

f,ax = plt.subplots(1,2, figsize=(10,5))

df_train[['Survived','Sex']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Sex vs Survived')



sns.countplot('Sex', hue ='Survived', data = df_train, ax = ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
sns.countplot('Pclass',hue='Survived', data = df_train)

plt.title('Pclass: Survived vs dead')

plt.show()
## Multivariate plots



pd.plotting.scatter_matrix(df_train, figsize =(10,10))

plt.show()
sns.violinplot(data = df_train, x ='Sex', y='Age')

plt.show()
f, ax= plt.subplots(1,2,figsize= (10,5))

sns.violinplot('Pclass','Age', hue='Survived', data = df_train, split= True, ax =ax[0])

ax[0].set_title("Pclass vs Age - Survived")

ax[0].set_yticks(range(0,110,10))



sns.violinplot('Sex','Age',hue='Survived', data= df_train, split=True, ax = ax[1])

ax[1].set_title('Sex vs Age - Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
##pairplot



sns.pairplot(data = df_train[['Fare','Survived','Pclass','Age']],

            hue='Survived', dropna = True)

plt.show()
# kde - Kernel Density estimation

sns.FacetGrid(df_train, hue='Survived', size =5).map(sns.kdeplot, 'Fare').add_legend()

plt.show()
sns.jointplot(x='Fare',y='Age', data =df_train)

plt.show()
##jointplot

sns.jointplot('Fare','Age',data = df_train, kind='reg')

plt.show()
##swarmplot

sns.swarmplot('Pclass','Age',data = df_train)
##Heatmap

plt.figure(figsize=(10,5))

sns.heatmap(df_train.corr(), annot= True, cmap ='cubehelix_r')

plt.show()

plt.imshow(df_train.corr(), cmap='hot', interpolation = 'nearest')

plt.show()
#barchart

df_train['Pclass'].value_counts().plot(kind='bar')
sns.factorplot('Pclass','Survived', hue='Sex', data =df_train)
sns.factorplot('SibSp','Survived',hue='Pclass', data= df_train)
#barplot and factor plot - the lines in the bar plot indicate the variance of the variable

f,ax =plt.subplots(1,2, figsize=(10,5))

sns.barplot('SibSp','Survived', data = df_train,ax = ax[0])

ax[0].set_title('SipSp vs Survived in barplot')



sns.factorplot('SibSp', 'Survived', data = df_train, ax = ax[1])

ax[1].set_title('Sibsp vs Survived in factorplot')

plt.close(2)

plt.show()

#distplot - distribution plot of univariate variable

f,ax = plt.subplots(1,3,figsize = (10,5))

sns.distplot(df_train[df_train['Pclass']==1]['Fare'], ax = ax[0])

ax[0].set_title('Pclass 1  - Fare')

sns.distplot(df_train[df_train['Pclass']==2]['Fare'], ax =ax[1])

ax[1].set_title('Pclass 2 - Fare')

sns.distplot(df_train[df_train['Pclass']==3]['Fare'], ax = ax[2])

ax[2].set_title('Pclass 3 - Fare')

plt.show()
df_train.shape, df_train.size # size = columns*rows
## check missing data

df_train.isnull().sum()
def check_missing_data(df):

    flag = df.isnull().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum() )/ df.isnull().count()*100

        output = pd.concat([total, percent], axis =1, keys=['Total','Percent'])

        data_type =[]

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Dtypes'] = data_type

        return np.transpose(output)

    else:

        return False

        
check_missing_data(df_train)
check_missing_data(df_test)
df_train.info()
df_train['Age'].unique()
# to view random 5 rows

df_train.sample(5)
df_train.describe()
df_train.isna().sum()
df_train.isnull().sum()
df_train.groupby('Survived').count()
df_train.where(df_train['Age']==35).head(2)
df_train[df_train['Age']==30]
X = df_train.iloc[:,:-1].values

y = df_train.iloc[:,:-1].values
## Data cleaning



def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels = group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x : x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels = group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df



def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis = 1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



df_train =transform_features(df_train)

df_test = transform_features(df_test)

df_train.head()



## encoding features

def encode_features(df_train, df_test):

    features = ['Fare','Cabin','Age','Sex','Lname','NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing. LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test
df_train, df_test = encode_features(df_train, df_test)

df_train.head()
## train test split



x_all = df_train.drop(['Survived','PassengerId'], axis = 1)

y_all = df_train['Survived']

num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size = num_test, random_state = 100)
result = None
from sklearn.metrics import make_scorer

rfc = RandomForestClassifier()



parameters = {'n_estimators': [4,6,9],

             'max_features' : ['log2','sqrt','auto'],

              'criterion' : ['entropy','gini'],

              'max_depth' : [2,3,5,10],

              'min_samples_split' : [2,3,5],

              'min_samples_leaf' : [1,5,8]

             }

acc_scorer = make_scorer(accuracy_score)



## grid search

grid_obj = GridSearchCV(rfc, parameters, scoring = acc_scorer)

grid_obj.fit(X_train, y_train)



rfc = grid_obj.best_estimator_

rfc.fit(X_train, y_train)
## prediction

rfc_prediction = rfc.predict(X_test)

rfc_score = accuracy_score(y_test, rfc_prediction)

print(rfc_score)
## XGBoost

from xgboost import XGBClassifier

xgboost = XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05).fit(X_train,y_train)
#prediction

xgb_prediction = xgboost.predict(X_test)

xgb_score= accuracy_score(y_test, xgb_prediction)

print(xgb_score)
## Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

logreg_score =accuracy_score(y_test, logreg_pred)

print(logreg_score)
## Decision tree classifier



from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state = 1)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_score = accuracy_score(y_test, dt_pred)

print(dt_score)
## Extra Tree classifier

from sklearn.tree import ExtraTreeClassifier



etc = ExtraTreeClassifier()

etc.fit(X_train, y_train)
etc_pred = etc.predict(X_test)

etc_score = accuracy_score(y_test, etc_pred)

print(etc_score)
##Submission

X_train = df_train.drop("Survived",axis=1)

y_train = df_train["Survived"]
X_train = X_train.drop("PassengerId",axis=1)

X_test  = df_test.drop("PassengerId",axis=1)
xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
Y_pred = xgboost.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)