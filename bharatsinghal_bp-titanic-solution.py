# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow import keras

# putting the training and test data in variables
Training_set = pd.read_csv("/kaggle/input/titanic/train.csv")
Test_set = pd.read_csv("/kaggle/input/titanic/test.csv")

Training_set["Cabin_Class"]="NA"
Test_set["Cabin_Class"]="NA"
Training_set["Cabin_No"]=0
Test_set["Cabin_No"]=0



T_set = len(Training_set)

#asdf = 2
#print(len(Training_set["Cabin"][1][0:asdf]))

# A VERY INEFFICIENT WAY TO CODING AND ADDING NEW COLUMNS
for i in range(T_set):
    if pd.isnull(Training_set["Cabin"][i]) :
        Training_set["Cabin_Class"][i] = "NA"
    else :
        Training_set["Cabin_Class"][i]=Training_set["Cabin"][i][0]

for i in range(T_set):
    if pd.isnull(Training_set["Cabin"][i]) :
        Training_set["Cabin_No"][i] = 0
    else :
        #ssdf = len(Test_set["Cabin"][i])
        sdf =3
        if str.isnumeric(Training_set["Cabin"][i][1:sdf]):
            Training_set["Cabin_No"][i]=int(Training_set["Cabin"][i][1:sdf])
        else :
            Training_set["Cabin_No"][i] = 0
            
#print(Test_set["Cabin"][0:3])

Tr_mean = sum(Training_set["Cabin_No"])/T_set

for i in range(T_set):
    if Training_set["Cabin_No"][i] == 0 :
        Training_set["Cabin_No"][i]=Tr_mean
    


Te_set = len(Test_set)

for i in range(Te_set):
    if pd.isnull(Test_set["Cabin"][i]) :
        Test_set["Cabin_Class"][i] = "NA"
    else :
        Test_set["Cabin_Class"][i]=Test_set["Cabin"][i][0]

for i in range(Te_set):
    if pd.isnull(Test_set["Cabin"][i]) :
        Test_set["Cabin_No"][i] = 0
    else :
        #sdf = len(Test_set["Cabin"][i])
        #print (i)
        sdf = 3
        if str.isnumeric(Test_set["Cabin"][i][1:sdf]):
            Test_set["Cabin_No"][i]=int(Test_set["Cabin"][i][1:sdf])
        else :
            Test_set["Cabin_No"][i] = 0

Te_mean = sum(Test_set["Cabin_No"])/Te_set
#print(Te_mean)

for i in range(Te_set):
    if Test_set["Cabin_No"][i] == 0 :
        Test_set["Cabin_No"][i]=Te_mean
    


y = Training_set["Survived"]        
#Test_set.head()
Training_set.head()


#y.head()
#print(Test_set["Sex"].unique())
#print(Training_set["Sex"].unique())

#print(Test_set["Cabin_Class"].unique())
#print(Training_set["Cabin_Class"].unique())

print(pd.crosstab(Training_set['Sex'], Training_set['Survived']))
# for col in Training_set: 
#    print(col) 

from sklearn import preprocessing

full_data = [Training_set, Test_set]
for dataset in full_data:
    dataset['Sex']= dataset['Sex'].map({'male': 1,'female': 0}).astype(int)
    dataset['Embarked']= dataset['Embarked'].fillna('S')
    #dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2,}).astype(int)
    #dataset['Age'] = preprocessing.scale(dataset['Age'])
    dataset["Pclass"] = dataset["Pclass"].astype("category")
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    dataset['Family Size'] = dataset['SibSp'] + dataset['Parch']+1
    dataset['Alone'] = dataset['Family Size'].apply(lambda x:1 if x==1 else 0)
    dataset['Has_Cabin'] = dataset['Cabin_Class'].apply(lambda x:0 if x=="NA" else 1)
    dataset['Cabin_Class'] = dataset['Cabin_Class'].map({'NA':0,'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'T':9}).astype("category")

# FIXING THE TITLE by using RegEx that allows us to extract     
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Mme'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Ms'], 'Mrs')

#print(pd.crosstab(Training_set['Title'], Training_set['Sex']))

features = ["Pclass" , "Sex" , "SibSp" , "Parch","Family Size" ,"Age","Fare", "Embarked","Has_Cabin","Cabin_Class","Alone","Title"]
#,"Cabin_No","Cabin_Class"

x = pd.get_dummies(Training_set[features])
X_test = pd.get_dummies(Test_set[features])

missing_val_count_by_column = (x.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column>0])

missing_val_count_by_column = (X_test.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column>0])

x.fillna(x.mean(), inplace=True)
X_test.fillna(x.mean(), inplace=True)

x, X_test = x.align(X_test,join='outer',axis=1)
#print(Test_set["Has_Cabin"].unique())
#print(Training_set["Has_Cabin"].unique())
#list(x)
x
X_test
#X_test.head()
#X_test.insert(15,"Cabin_Class_T",0)
#mapping the various factors to see which factors have highest correlation
colormap = plt.cm.RdBu
x_t = pd.concat([y,x],axis=1)
#x_t.head()
plt.figure(figsize=(10,10))
plt.title('Pearson Correl of variables',y=1.05,size=15)
sns.heatmap(x_t.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=False,fmt='.1f')
#g = sns.pairplot(x_t[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare',u'Family Size']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
#g.set(xticklabels=[])

z=0
for col in x: 
    z +=1

print(z)
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[z]),
    keras.layers.Dense(10),
    keras.layers.Dense(20,activation = tf.nn.softmax),
    keras.layers.Dense(1)
])


#,activation = tf.nn.softmax
model.compile(optimizer='adam',loss='mean_squared_error')
# other options are 'adam'; 'sgd'
model.fit(x,y,epochs=500)

pred = model.predict(X_test)
#print (pred[1][0])

length =len(pred) 
#print (length)

predictions = [None]*length

for i in range(length) :
    if pred[i][0]>0.5:
        dummy = 1
    else :
        dummy = 0
    predictions[i] = dummy
    
#print (predictions)

# writing code for tree classifiers so that their input can also be incorporated into this : LEARNING : CREATING ENSEMBLES

ntrain = x.shape[0]
ntest = X_test.shape[0]
#print(ntrain, ntest)
SEED = 0 
NFOLDS = 5 # data to be spliced into 5 parts as there are 5 tree classifiers that we are testing right now
kf = KFold(n_splits = NFOLDS, random_state = SEED)

# create a class to help faciliate the calculations for all classifiers at once
class Clubbedcode(object):
    def __init__(self,clf,seed=0,params=None):
        #params['random state']=seed
        self.clf=clf(**params)
#The self is used to represent the instance of the class. With this keyword, you can access the attributes and methods of the class in python. 
#It binds the attributes with the given arguments.    
    
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    
    def predict(self,x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)
# we need to define this program because we need to use it below

    def feature_importances(self,x,y):
        return(self.clf.fit(x,y).feature_importances_)

        
    
    
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#HAVE NOT BEEN ABLE TO UNDERSTAND IT COMPLETELY :(
    
    
# Prepare the data to be fed into our models
y_train = y.ravel()
#train = Training_set.drop(['Survived'], axis=1)
x_train =x.values # Creates an array of the train data
x_test = X_test.values # Creats an array of the test data

print(x_train.shape[:])

from sklearn.model_selection import RandomizedSearchCV # Number of trees in random forest
from sklearn.model_selection import GridSearchCV

rc_n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of features to consider at every split
rc_max_features = ['auto', 'sqrt']               # Maximum number of levels in tree
rc_max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
rc_max_depth.append(None) # Minimum number of samples required to split a node
rc_min_samples_split = [2, 5, 10] # Minimum number of samples required at each leaf node
rc_min_samples_leaf = [1, 2, 4] # Method of selecting samples for training each tree
rc_bootstrap = [True, False] # Create the random grid
rc_random_grid = {'n_estimators': rc_n_estimators,
               'max_features': rc_max_features,
               'max_depth': rc_max_depth,
               'min_samples_split': rc_min_samples_split,
               'min_samples_leaf': rc_min_samples_leaf,
               'bootstrap': rc_bootstrap}
print(rc_random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rc = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rc_random = RandomizedSearchCV(estimator = rc, param_distributions = rc_random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rc_random.fit(x_train, y_train)
rc_random.best_params_

#Hyper Paramater Tuning

rc_param_grid = {
    'bootstrap': [False],
    'max_depth': [50,55,60, 70, 80, 90],
    'max_features': ['auto'],
    'min_samples_leaf': [1,2,3 , 4, 5],
    'min_samples_split': [9, 10, 11],
    'n_estimators': [1500, 2000, 2500, 3000, 3500]
}

# Create a based model
#rc = RandomForestClassifier()
# Instantiate the grid search model
rc_grid_search = GridSearchCV(estimator = rc, param_grid = rc_param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

rc_grid_search.fit(x_train, y_train)
rc_grid_search.best_params_
ada_n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of features to consider at every split
ada_learning_rate = [.1,.25,.5,.75,1.0]
ada_random_grid = {'n_estimators': ada_n_estimators,
               'learning_rate': ada_learning_rate}
print(rc_random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
ada =AdaBoostClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
ada_random = RandomizedSearchCV(estimator = ada, param_distributions = ada_random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
ada_random.fit(x_train, y_train)
ada_random.best_params_

'''ada_param_grid = {
    'learning_rate': [0.09,.08,.1,.11,.12],
    'n_estimators': [180, 200, 210 ,220, 240]
}

# Create a based model
# Instantiate the grid search model
ada_grid_search = GridSearchCV(estimator = ada, param_grid = ada_param_grid, cv = 3, n_jobs = -1, verbose = 2)

ada_grid_search.fit(x_train, y_train)
ada_grid_search.best_params_'''
svc_kernel = ['linear','polynomial','rbf','sigmoid'] 
svc_C = [float(x) for x in np.linspace(start = 0.01, stop = 0.2, num = 20 )]
svc_random_grid = {'kernel': svc_kernel,
               'C': svc_C}
print(svc_random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
svc =SVC()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
svc_random = RandomizedSearchCV(estimator = svc, param_distributions = svc_random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
svc_random.fit(x_train, y_train)
svc_random.best_params_
'''svc_C = [float(x) for x in np.linspace(start = 0.01, stop = 0.05, num = 20 )]
svc_param_grid = {
    'kernel': ['linear'],
    'C': svc_C
}

# Create a based model
# Instantiate the grid search model
svc_grid_search = GridSearchCV(estimator = svc, param_grid = svc_param_grid, cv = 3, n_jobs = -1, verbose = 2)

svc_grid_search.fit(x_train, y_train)
svc_grid_search.best_params_'''
gb_n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of features to consider at every split
gb_max_depth = [int(x) for x in np.linspace(1, 11, num = 11)]
gb_min_samples_leaf = [1, 2, 4] # Method of selecting samples for training each tree
gb_random_grid = {'n_estimators': gb_n_estimators,
               'max_depth': gb_max_depth,
               'min_samples_leaf': gb_min_samples_leaf,}
print(gb_random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
gb = GradientBoostingClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = gb_random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
gb_random.fit(x_train, y_train)
gb_random.best_params_

'''gb_param_grid = {
    'n_estimators': [1100,1200,1300,1400,1500,1600,1250],
    'min_samples_leaf': [1,2,4,8],
    'max_depth': [2]
}

# Create a based model
# Instantiate the grid search model
gb_grid_search = GridSearchCV(estimator = gb, param_grid = gb_param_grid, cv = 3, n_jobs = -1, verbose = 2)

gb_grid_search.fit(x_train, y_train)
gb_grid_search.best_params_'''
et_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)] # Number of features to consider at every split
et_max_depth = [int(x) for x in np.linspace(1, 11, num = 11)]
et_min_samples_leaf = [1, 2, 4] # Method of selecting samples for training each tree
et_random_grid = {'n_estimators': et_n_estimators,
               'max_depth': et_max_depth,
               'min_samples_leaf': et_min_samples_leaf,}
print(et_random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
et = ExtraTreesClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
et_random = RandomizedSearchCV(estimator = et, param_distributions = et_random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
et_random.fit(x_train, y_train)
et_random.best_params_
'''et_param_grid = {
    'n_estimators': [1000,1050,1100,1150,1200,1250,1300],
    'min_samples_leaf': [1,2,4,8],
    'max_depth': [1,2,3,4,5,6,7,8]
}

# Create a based model
# Instantiate the grid search model
et_grid_search = GridSearchCV(estimator = et, param_grid = et_param_grid, cv = 3, n_jobs = -1, verbose = 2)

et_grid_search.fit(x_train, y_train)
et_grid_search.best_params_'''
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 600,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 60,
    'min_samples_leaf': 5,
    #'min_samples_split':9,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':1000,
    #'max_features': 0.5,
    'max_depth': 5,
    'min_samples_leaf': 1,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 180,
    'learning_rate' : 0.09
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 1100,
     #'max_features': 0.2,
    'max_depth': 2,
    'min_samples_leaf': 8,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.031
    }
# Create 5 objects that represent our 4 models
rf = Clubbedcode(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = Clubbedcode(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = Clubbedcode(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = Clubbedcode(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = Clubbedcode(clf=SVC, seed=SEED, params=svc_params)
# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

print (rf_feature,et_feature,ada_feature,gb_feature)


print(rf_feature.shape[:],et_feature.shape[:],ada_feature.shape[:],gb_feature.shape[:])

cols = x.columns.values
print (cols)
print (cols.shape[:])

# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })
# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        #size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(10)
y_plot = feature_dataframe['mean'].values
x_plot = feature_dataframe['features'].values
data = [go.Bar(
            x= x_plot,
             y= y_plot,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')
keras_pred = model.predict(x)

k_len = len(keras_pred)

#print(k_len)

predict = [None]*k_len

for i in range(k_len) :
    if keras_pred[i][0]>0.5:
        dummy = 1
    else :
        dummy = 0
    predict[i] = dummy

#print(predict)
    
#print(predictions)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
        'Keras':predict
    })
#print(base_predictions_train)
#base_predictions_train.head()
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')
#print(keras_pred[0]) 
#keras_pred[0].apply(lambda x:1 if x>0.5 else 0)
#print(keras_pred[0]) 

zu = np.array(predictions).reshape(-1,1)
#print (zu)
#print (predictions)
#print(et_oof_train)
cva = np.array(predict).reshape(-1,1)

x2_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train,cva), axis=1)
x2_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test,zu), axis=1)

#x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train,keras_pred), axis=1)
#x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test,pred), axis=1)


#x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
#x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

print(x2_train)
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x2_train, y)
gbm_predictions = gbm.predict(x2_test)

#print(gbm_predictions)

#print (y)
#print (Test_set.PassengerId)
output = pd.DataFrame({'PassengerId': Test_set.PassengerId, 'Survived': gbm_predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
#x.to_csv('X.csv', index=False)
#print("Your submission was successfully saved!")