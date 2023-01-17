



# # color palletes

# male_female_pal = ['#3489d6', '#e64072']

# survival_pal = ['#2a2a2a', '#ff0000']

# sns.set_palette(survival_pal)

# sns.set_style("whitegrid")



# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, LabelBinarizer, scale, Normalizer, PowerTransformer, MaxAbsScaler

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# from sklearn.svm import SVC, NuSVC, LinearSVC

# from sklearn.linear_model import LogisticRegression

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.neighbors import KNeighborsClassifier



# import lightgbm as lgb



# import eli5

# from eli5.sklearn import PermutationImportance
# numerical analysis

import numpy as np

# storing and processing in dataframes

import pandas as pd



# basic plotting

import matplotlib.pyplot as plt

# advanced plotting

import seaborn as sns



# splitting dataset into train and test

from sklearn.model_selection import train_test_split

# scaling features

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, LabelEncoder

# selecting important features

from sklearn.feature_selection import RFECV

# k nearest neighbors model

from sklearn.neighbors import KNeighborsClassifier

# accuracy

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
# plot style

sns.set_style('whitegrid')



# color palettes

male_female_pal = ['#3489d6', '#e64072']

survival_pal = ['#2a2a2a', '#ff0000']

sns.set_palette(survival_pal)
# get training dataset

train = pd.read_csv('../input/train.csv')



# first few rows of train dataset

train.head()
# get test dataset

test = pd.read_csv('../input/test.csv')



# first few rows of test dataset

test.head()
# no. of rows and columns

train.shape
#  columns names

train.columns
# consise summary of dataframe

train.info()
# descriptive statistics

train.describe(include='all')
def get_missing_vals_info(df):

    '''get no. of missing values information'''

    

    # no. of missing values in each column of the dataframe

    print(df.isna().sum())

    

    # visualizing missing values in each column

    

    # plot figure

    plt.figure(figsize=(12, 6))

    # plot missing values heatmap

    sns.heatmap(df.isna(), cbar=False, cmap='cividis')

    # title

    plt.title('Missing values in each columns')

    # show the plot

    plt.show()
# missing values in train dataset

get_missing_vals_info(train)
# missing values in train dataset

get_missing_vals_info(test)
# Class distribution



plt.figure(figsize=(4, 5))

sns.countplot(x='Survived', data=train)

plt.show()
# How being in different categories resulted in the survival ?



cat_cols = ['Pclass', 'Sex', 'Embarked']



fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for ind, val in enumerate(cat_cols):

    sns.countplot(x=val, hue='Survived', data=train, ax=ax[ind])

    ax[ind].legend(['Did not survived', 'Survived'])
# Did people hold on to their families ?



cat_cols = ['SibSp', 'Parch']



fig, ax = plt.subplots(1, 2, figsize=(15, 4))

for ind, val in enumerate(cat_cols):

    sns.countplot(x=val, hue='Survived', data=train, ax=ax[ind])

    ax[ind].legend(['Did not survived', 'Survived'])
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

for ind, col in enumerate(['Age', 'Fare']):

    ax[ind] = sns.kdeplot(train.loc[train['Survived']==0, col].dropna(), shade=True, ax=ax[ind])

    ax[ind] = sns.kdeplot(train.loc[train['Survived']==1, col].dropna(), shade=True, ax=ax[ind])

    ax[ind].set_xlabel(col)

    ax[ind].legend(['Did not survived', 'Survived'])
# Correlation between columns



plt.figure(figsize=(8, 6))

df_corr = train.drop('PassengerId', axis=1).corr()

sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu', vmax=0.8, vmin=-0.8)

plt.show()
# Pairplot



plt.figure(figsize=(7, 7))

sns.pairplot(train.drop('PassengerId', axis=1), hue="Survived", palette=survival_pal)

plt.plot()
# Filling Embarked with most frequent value

# =========================================



print(train['Embarked'].value_counts())

most_freq = train['Embarked'].value_counts().index[0]

train['Embarked'].fillna(most_freq, inplace=True)
# Filling age with respect to title

# ==================================



# extracting the title

train["Title"] = train["Name"].str.extract('([A-Za-z]+)\.',expand=False)

test["Title"] = test["Name"].str.extract('([A-Za-z]+)\.',expand=False)



# replacing similar titles

for i in [train, test]:

    i['Title'] = i['Title'].replace('Mr', 'Mr')

    i['Title'] = i['Title'].replace(('Mme', 'Ms'), 'Mrs')

    i['Title'] = i['Title'].replace('Mlle', 'Miss')

    i['Title'] = i['Title'].replace(('Capt', 'Col', 'Major', 'Dr','Rev'), 'Officer')

    i['Title'] = i['Title'].replace(('Jonkheer', 'Don', 'Sir', 'Countess','Dona', 'Lady'), 'Royalty')
# Title vs Age Distribution



sns.set_palette('Paired')

plt.figure(figsize=(15, 6))



ax = sns.kdeplot(train[train['Title']=='Mr']['Age'], shade=True, label='Mr')

ax = sns.kdeplot(train[train['Title']=='Mrs']['Age'], shade=True, label='Mrs')

ax = sns.kdeplot(train[train['Title']=='Miss']['Age'], shade=True, label='Miss')

ax = sns.kdeplot(train[train['Title']=='Master']['Age'], shade=True, label='Master')

ax = sns.kdeplot(train[train['Title']=='Officer']['Age'], shade=True, label='Officer')



ax.set_xlim(-10, 90)

ax.set_xlabel('Age')

ax.set_title('Distribution of Age of based on Title')

plt.show()
# fill age wrt title

for df in [train, test]:

    for title in df['Title'].unique():

        age = df.loc[df['Title']==title, 'Age'].mean()

        df[df['Title']==title].fillna(age, inplace=True)
train.info()
# converting to catogorical values into categorical columns 



cat_cols = ['Pclass', 'Sex', 'Embarked']

for i in cat_cols:

    train[i] = train[i].astype('category')

    test[i] = test[i].astype('category')

    

# train.info()
# Class - Gender - Survival



g = sns.FacetGrid(train, col='Embarked', size=4)

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1, 2, 3], 

      hue_order=['male', 'female'], palette=male_female_pal)

g.add_legend()

plt.show()
plt.figure(figsize=(10, 6))

sns.countplot(x="Embarked", hue="Title", data=train)

plt.show()
plt.figure(figsize=(10, 6))

sns.countplot(x="Pclass", hue="Title", data=train)

plt.show()
# Mean of of age wrt Title



tr = train[['Age', 'Title']]

ts = test[['Age', 'Title']]

tr_ts = pd.concat([tr, ts])



print(tr_ts.groupby('Title').mean())
# plt.figure(figsize=(60,5))

# ax = sns.countplot(x='Age', hue='Survived', data=train)

# plt.legend(['Not Survived', 'Survived'])

# plt.show()



# plt.figure(figsize=(200, 5))

# ax = sns.countplot(x='Fare', hue='Survived', data=train)

# plt.legend(['Not Survived', 'Survived'])

# plt.show()
# Binning Age and Fare



train['age_cat'] = pd.cut(train['Age'], 

                          bins = [0, 0.99, 7, 23, 58, 100],

                          labels = ["infant", "child", "young", "adult", "senior"],

                          include_lowest=True)

test['age_cat'] = pd.cut(test['Age'], 

                         bins = [0, 0.99, 7, 23, 58, 100],

                         labels = ["infant", "child", "young", "adult", "senior"],

                         include_lowest=True)



train['fare_cat'] = pd.cut(train['Fare'], 

                           bins = [0, 12, 40, 80, 1000],

                           labels = ['least', 'low', 'mid', 'high'],

                           include_lowest=True)

test['fare_cat'] = pd.cut(test['Fare'], 

                           bins = [0, 12, 40, 80, 1000],

                           labels = ['least', 'low', 'mid', 'high'],

                           include_lowest=True)
# Extracting Cabin Type from Cabin name



c_train_type = train['Cabin'].str[0]

train['c_type'] = c_train_type

train['c_type'] = train['c_type'].fillna('unknown')



c_test_type = test['Cabin'].str[0]

test['c_type'] = c_test_type

test['c_type'] = test['c_type'].fillna('unknown')
# Age, Fare, Cabin category vs Survival



fig, ax = plt.subplots(1, 3, figsize=(24, 5))

for ind, val in enumerate(['age_cat', 'fare_cat', 'c_type']):

    sns.countplot(x=val, hue='Survived', data=train, ax=ax[ind])
# Family member count and Family Size and is alone



train['fam_count'] = train['SibSp']+train['Parch']

test['fam_count'] = test['SibSp']+test['Parch']



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



train['fam_size'] = train['fam_count'].map(size)

test['fam_size'] = test['fam_count'].map(size)



train['is_alone'] = train['fam_size']=='alone'

test['is_alone'] = test['fam_size']=='alone'
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

for ind, val in enumerate(['fam_count', 'fam_size', 'is_alone']):

    sns.countplot(x=val, hue='Survived', data=train, ax=ax[ind])
# dataframe



train.head()
# Scaling Age and Fare



mm = MinMaxScaler()

for i in ['Age', 'Fare']:

    train[i] =  mm.fit_transform(train[i].values.reshape(-1,1))

    test[i] =  mm.fit_transform(test[i].values.reshape(-1,1))
# Label Binerizer Sex



lb = LabelBinarizer()

for i in ['Sex', 'is_alone']:

    train[i] =  lb.fit_transform(train[i])

    test[i] =  lb.fit_transform(test[i])
# Label Encoding Pclass



en = LabelEncoder()

train['Pclass'] =  en.fit_transform(train['Pclass'])

test['Pclass'] =  en.fit_transform(test['Pclass'])
# Create dummies for nominal categorical columns



def create_dummies(df, column_name):

    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df



# for i in ['Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'age_cat', 'fare_cat', 'c_type', 'fam_count', 'fam_size']:

#     train = create_dummies(train, i)

#     test = create_dummies(test, i)
# Droping columns



train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# dataframe



train.head()
# Final correlation heatmap



plt.figure(figsize=(10, 8))

sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True, fmt='.1f', cmap='RdBu', vmax=0.8, vmin=-0.8)

plt.show()
# X = train.drop(['Survived', 'PassengerId'], axis=1)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'fam_count', 'is_alone']

X = train[features]

y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# naive bayes



nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy_score(y_pred, y_test)

#print(classification_report(y_pred, y_test))
plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')

plt.show()
perm = PermutationImportance(nb, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# Logistic regression



lr = LogisticRegression(C = 1, penalty= 'l2', solver= 'liblinear')

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy_score(y_pred, y_test)
#print(classification_report(y_pred, y_test))

plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')

plt.show()
perm = PermutationImportance(lr, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# svm



model = SVC()



hyperparameters = {

    'C': [0.1, 1, 10, 100],

    'gamma': [1, 0.1, 0.01],

    'kernel': ['rbf', 'linear']

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



svc = grid.best_estimator_

y_pred = svc.predict(X_test)



print(grid.best_params_)

print(grid.best_estimator_)

print(grid.best_score_)
#print(classification_report(y_pred, y_test))

plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')

plt.show()
perm = PermutationImportance(svc, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# K Nearest Neighbours



model = KNeighborsClassifier()



hyperparameters = {

    "n_neighbors" : range(1,20,2),

    'weights' : ['uniform', 'distance'],

    'p' : [1, 2]

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



knn = grid.best_estimator_

y_pred = knn.predict(X_test)



print(grid.best_params_)

print(grid.best_estimator_)

print(grid.best_score_)
#print(classification_report(y_pred, y_test))

plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')

plt.show()
perm = PermutationImportance(knn, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# Decision Tree



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy_score(y_pred, y_test)

#print(classification_report(y_pred, y_test))
from sklearn import tree

import graphviz 



dot_data = tree.export_graphviz(dt, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris")



dot_data = tree.export_graphviz(dt, out_file=None, 

                     feature_names=X_train.columns,  

                     class_names=['Survived', 'Not Survived'],  

                     filled=True, rounded=True,  

                     special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [3, 5, 7, 10],

                   "max_features": ["log2", "sqrt", 'auto'], 

                   'min_samples_leaf' : [2, 3, 4, 5],

                   'min_samples_split' : [2, 3, 4, 5]

}



grid = GridSearchCV(dt, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



dt = grid.best_estimator_

y_pred = dt.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)



#print(classification_report(y_pred, y_test))

plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')



perm = PermutationImportance(dt, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# Random Forest



model = RandomForestClassifier()



hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   'min_samples_leaf' : [2, 3, 4, 5],

                   'min_samples_split' : [2, 3, 4, 5],

                   "n_estimators": [6, 9]

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



rf = grid.best_estimator_

y_pred = rf.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)



#print(classification_report(y_pred, y_test))

plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_pred, y_test), annot=True, cbar=False, fmt='1d', cmap='Blues')



perm = PermutationImportance(rf, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# test.head()
# test.isnull().sum()
holdout_ids = test["PassengerId"]

holdout_features = test[features]

holdout_predictions = lr.predict(holdout_features)



submission = pd.DataFrame({"PassengerId": holdout_ids, 

                           "Survived": holdout_predictions})

print(submission.head())



submission.to_csv("submission.csv",index=False)