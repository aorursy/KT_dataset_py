# Importing required libraries



# Handle table-like data and matrices

import pandas as pd

import numpy as np



import re

import sklearn

import xgboost as xgb



# Visualization 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls 



#Ignore warnings 

import warnings

warnings.filterwarnings('ignore')



#Base models for stacking



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier



# Modelling Helpters

from sklearn.preprocessing import Imputer, Normalizer, scale

from sklearn.cross_validation import KFold

from sklearn.feature_selection import RFECV

# Load in the train and test datasets



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



PassengerId = test['PassengerId']
# Take a look at the dataset

train.head()
full_data = [train,test]



# Feature telling whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x)==float else 1)

test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)



# Feature: Family_Size as a combination of SibSp and Parch

for dataset in full_data:

    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['Family_Size', 'Survived']].groupby(['Family_Size'], as_index=False).mean())

print ("\n")



# Feature: Is_Alone from FamilySize

for dataset in full_data:

    dataset['Is_Alone']=0

    dataset.loc[dataset['Family_Size']==1,'Is_Alone'] = 1

print(train[['Is_Alone','Survived']].groupby(['Is_Alone'],as_index=False).mean())

print("\n")



# Cleaning missing values

# from Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean())

print("\n")



# from Fare column and create new feature Categorical_Fare

for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())

train['Categorical_Fare'] = pd.qcut(train['Fare'],4)

print(train[['Categorical_Fare','Survived']].groupby(['Categorical_Fare'],as_index = False).mean())

print("\n")

# Categorical_Age: There are plenty of missing values in this feature => need to generate random numbers between mean + std and mean - std



for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count =dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std,age_avg+age_std,size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['Categorical_Age'] = pd.qcut(train['Age'],5)

print(train[['Categorical_Age','Survived']].groupby(['Categorical_Age'],as_index=False).mean())



print("\n")    





# Processing names

# Creating function to extract names

def get_title(name):

    title_search = re.search('([A-Za-z]+)\.',name)

    # If the title exists, extract and return it

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())





for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} )#.astype(int)

    

    

    # Mapping titles

    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master" :4, "Rare":5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S':0,'C':1,'Q':2})#.astype(int)

    

    # Mapping Fare

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

    dataset['Age'] = dataset['Age'].astype(int)

    



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['Categorical_Age', 'Categorical_Fare'],axis=1)



test = test.drop(drop_elements, axis = 1)







train.head(3)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y = 1.05, size = 15)

sns.heatmap(train.astype(float).corr(),linewidth =0.1, vmax = 1.0, square= True, cmap = colormap, linecolor = 'white', annot = True)
# parameters

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)



# SklearnHelper class to extend sklearn classifier

class SklearnHelper(object):

    '''

    Starting with def init: Python standard for invking the default constructor for the class. This measn

    when you want to create an object(classifier), you have to give it the parameters of clf (the sk clf you want), 

    seed (random seed) and params (parameters for the classifier)'''

    

    def __init__(self, clf, seed = 0, params = None):

        params['random_state'] = seed

        self.clf = clf(**params)

        

    def fit(self, x , y):

        return self.clf.fit(x,y)

    

    def predict(self,x):

        return self.clf.predict(x)

    

    def feature_importances(self, x, y):

        return (self.clf.fit(x,y).feature_importances_)

        



def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))

    

    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.fit(x_tr, y_tr)

        

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i,:] = clf.predict(x_test)

        

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

# Random Forest paarameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

    'warm_start':True,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose':0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators': 500,

    'max_depth':8,

    'min_samples_leaf': 2,

    'verbose':0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate': 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose' : 0

}



# Support Vector Classifier parameters

svc_params = {

    'kernel': 'linear',

    'C' : 0.025

}
# Creating objsects to present the 5 models

rf = SklearnHelper(clf = RandomForestClassifier, seed = SEED, params= rf_params)

et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params= et_params)

ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params= ada_params)

gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params= gb_params)

svc = SklearnHelper(clf = SVC, seed = SEED, params= svc_params)
# Creating numpy arrays of train, test and target dataframes to feed into the models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis = 1)

x_train = train.values

x_test = test.values 

rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)

gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)

svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)



print("Training is complete")

rf_feature = rf.feature_importances(x_train, y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train, y_train)





cols = train.columns.values



# Create a dataframe with features

feature_dataframe = pd.DataFrame({'features':cols,

                                 'Random Forest feature importances': rf_feature,

                                 'Extra Trees feature importances': et_feature,

                                 'AdaBoost feature importances': ada_feature,

                                 'Gradient Boosting feature importances': gb_feature

                                 })
# Scatter plot for Random Forest feature importances



trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker = dict(

        sizemode= 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale = 'Portland',

        showscale = True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout = go.Layout(

    autosize = True,

    title = 'Random Forest Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go. Figure(data=data,layout=layout)

py.iplot(fig,filename = 'scatter2010')



# Scatter plot for Extra Trees feature importances

trace = go.Scatter(

    y = feature_dataframe['Extra Trees feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker = dict(

        sizemode= 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Extra Trees feature importances'].values,

        colorscale = 'Portland',

        showscale = True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout = go.Layout(

    autosize = True,

    title = 'Extra Trees Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go. Figure(data=data,layout=layout)

py.iplot(fig,filename = 'scatter2010')





# Scatter Plot for AdaBoost

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker = dict(

        sizemode= 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale = 'Portland',

        showscale = True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout = go.Layout(

    autosize = True,

    title = 'AdaBoost Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go. Figure(data=data,layout=layout)

py.iplot(fig,filename = 'scatter2010')





# Scatter Plot for Gradient Boost

trace = go.Scatter(

    y = feature_dataframe['Gradient Boosting feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker = dict(

        sizemode= 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Gradient Boosting feature importances'].values,

        colorscale = 'Portland',

        showscale = True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout = go.Layout(

    autosize = True,

    title = 'Gradient Boosting Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go. Figure(data=data,layout=layout)

py.iplot(fig,filename = 'scatter2010')
# Creating a new column containing the average of values row-wise



feature_dataframe['mean'] = feature_dataframe.mean(axis=1)

feature_dataframe.head(3)
#Plot the mean feature importances across all our classifiers into a Plotly bar plot



y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x = x,

             y = y,

            width = 0.5,

            marker =dict(

                color = feature_dataframe['mean'].values,

            colorscale = 'Portland',

            showscale=True,

            reversescale = False

            ),

            opacity = 0.6

         )]



layout = go.Layout(

    autosize=True,

    title = 'Barplots of Mean Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2,

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout = layout)

py.iplot(fig, filename = 'bar-direct-labels')

    
base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),

                                      'ExtraTrees': et_oof_train.ravel(),

                                      'AdaBoost': ada_oof_train.ravel(),

                                      'GradientBoost': gb_oof_train.ravel()

                                      })

base_predictions_train.head()
# Correlation Heatmap of the Second level training set



data = [

    go.Heatmap(

    z = base_predictions_train.astype(float).corr().values,

    x = base_predictions_train.columns.values,

    y = base_predictions_train.columns.values,

    colorscale = 'Portland',

    showscale=True,

    reversescale=True

    )

]

py.iplot(data, filename='labelled-heatmap')

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train),axis = 1)

x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test),axis=1)
gbm = xgb.XGBClassifier(

n_estimator = 2000,

max_depth = 4,

min_child_weight = 2,

gamma = 0.9,

subsample = 0.8,

colsample_bytree=0.8,

objective = 'binary:logistic',

nthread=-1,

scale_pos_weight=1)

gbm.fit(x_train,y_train)

predictions = gbm.predict(x_test)
print (gbm.score(x_train, y_train),gbm.score(x_test,predictions))
# Generating Submission File



StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,

                                  'Survived':predictions})

StackingSubmission.to_csv("StackingSubmission.csv", index=False)