# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#import plotly.offline as py

#py.init_notebook_mode(connected=True)

#import plotly.graph_objs as go

#import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import os

print(os.listdir("../input"))
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;



for dataset in full_data:

    scalar = StandardScaler()

    dataset['Name_length'] = scalar.fit_transform(dataset[['Name_length']])
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
test.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
def create_train_test_datasets(train, test, full_dataset):

    if full_dataset == 'Y':

        X_train = train.drop(['Survived'], axis=1)

        y_train = train['Survived']

        X_test = test.copy()

        y_test = pd.Series()

        

    else:

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_indx, val_idx in split.split(train, train['Survived']):

            strat_train_set = train.iloc[train_indx]

            strat_val_set   = train.iloc[val_idx]

        X_train = strat_train_set.drop(['Survived'], axis=1)

        y_train = strat_train_set['Survived']

        X_test = strat_val_set.drop(['Survived'], axis=1)

        y_test = strat_val_set['Survived']

    return X_train, y_train, X_test, y_test
#create the datasets for training and testing

X_train, y_train, X_test, y_test = create_train_test_datasets(train,test,'Y')
#check the shape of the datasets

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#submit predictions

#def submit_predictions(filename, predictions):

#    Submission = pd.DataFrame({'PassengerId': PassengerId,

#                            'Survived': predictions})

#    Submission.to_csv(filename, index=False)
# This function needs to be run only when using full dataset for Voting classifier prediction

def predict_soft_voting_full(X_train, y_train, X_test):

    lr_clf = LogisticRegression(random_state=42)

    rf_clf = RandomForestClassifier(random_state=42)

    gb_clf = GradientBoostingClassifier(random_state=42)

    svm_clf = SVC(probability=True, random_state=42)

    xgb_clf = XGBClassifier(random_state=42)

    voting_clf = VotingClassifier(

                estimators = [('lr',lr_clf),('rf',rf_clf),('gb',gb_clf),('svc',svm_clf),('xgb',xgb_clf)],

                voting='soft')

    voting_clf.fit(X_train, y_train)

    predictions = voting_clf.predict(X_test)

    #submit_predictions("soft_voting_full.csv",predictions)

predict_soft_voting_full(X_train, y_train, X_test)
# This function needs to be run when using validtion set for Voting classifier 

def predict_soft_voting_validation(X_train, y_train, X_test, y_test):

    lr_clf = LogisticRegression(random_state=42)

    rf_clf = RandomForestClassifier(random_state=42)

    gb_clf = GradientBoostingClassifier(random_state=42)

    svm_clf = SVC(probability=True, random_state=42)

    xgb_clf = XGBClassifier(random_state=42)

    voting_clf = VotingClassifier(

                estimators = [('lr',lr_clf),('rf',rf_clf),('gb',gb_clf),('svc',svm_clf),('xgb',xgb_clf)],

                voting='soft')

    voting_clf.fit(X_strat_train, y_strat_train)

    for clf in (lr_clf, rf_clf, gb_clf,svm_clf, xgb_clf, voting_clf):

        clf.fit(X_strat_train, y_strat_train)

        y_pred = clf.predict(X_strat_val)

        print(clf.__class__.__name__, accuracy_score(y_strat_val, y_pred))
lr_parms = {

'C' : np.logspace(-5, 8, 15),

'penalty': ['l1', 'l2']

}



# Random Forest parameters

rf_params = {

    'n_estimators': [250, 300,350],

     #'max_features': 0.2,

    'max_depth': [2, 4, 6],

    'max_features' : [2,4, 6]

}



# Extra Trees Parameters

et_params = {

    'n_estimators':[50,100,200],

    'criterion':['gini','entropy'],

    'max_depth': [30,40,50],

    'min_samples_leaf': [1,2,4]

}



# AdaBoost parameters

ada_params = {

    'n_estimators': [100,200,300],

    'learning_rate' : [0.01,0.03,0.1,0.5]

}



# Gradient Boosting parameters

gb_params = {

    "learning_rate": [0.01, 0.03,0.1, 0.3, 1.0],

    #"min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf": [4,8,12],

    "max_depth":[1,3,5,8],

    #"max_features":["log2","sqrt"],

    #"criterion": ["friedman_mse",  "mae"],

    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":[100,300,500]

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : ['linear','poly','rbf'],

    'C' : [0.05,0.10,0.30, 1.0]

    }



#xgb_params = {

# 'learning_rate' :[0.03,0.10, 0.30],

# 'n_estimators': [100,200,300],

# 'max_depth':[3,5,7,10,13],

# 'min_child_weight':[1,3,5,7],

# 'gamma' : [0.03, 0.1,0.3,1.0],

 #'subsample':0.8,

 #'colsample_bytree':0.8,

 #'reg_alpha':0.005,

 #'objective': 'binary:logistic',

 #'nthread':4,

 #'scale_pos_weight':1,

 #'seed':27

#}
def classifiers_predict(X_train, y_train, X_test, y_test, full_dataset, classifiers):

    

    train_stack_df = pd.DataFrame()

    test_stack_df = pd.DataFrame()

    

    if 'lr' in classifiers:

        #logistic Regression

        print("Logistic Regression Training Starts...")

        lr = LogisticRegression()

        grid_search_lr = GridSearchCV(lr, lr_parms, cv=5)

        grid_search_lr.fit(X_train, y_train)

        print(grid_search_lr.best_params_)

        print("Logistic Regression Training Ends...")

        

    

    if 'rf' in classifiers:    

        #Random Forest

        print("Random Forest Training Starts...")

        rf = RandomForestClassifier()

        grid_search_rf = GridSearchCV(rf, rf_params, cv=5)

        grid_search_rf.fit(X_train, y_train)

        print(grid_search_rf.best_params_)

        print("Random Forest Training Ends...")

    

    if 'et' in classifiers:

        #Extra Trees Classifier

        print("ExtraTree Classifier Training Starts...")

        et = ExtraTreesClassifier()

        grid_search_et = GridSearchCV(et, et_params, cv=5)

        grid_search_et.fit(X_train, y_train)

        print(grid_search_et.best_params_)

        print("ExtraTree Classifier Training Ends...")

    

    if 'ada' in classifiers:

        #AdaBoost Classifier

        print("AdaBoost Classifier Training Starts...")

        ada = AdaBoostClassifier()

        grid_search_ada = GridSearchCV(ada, ada_params, cv=5)

        grid_search_ada.fit(X_train, y_train)

        print(grid_search_ada.best_params_)

        print("AdaBoost Classifier Training Ends...")

    

    if 'gb' in classifiers:

        #GradientBoost Classifier

        print("GradientBoost Classifier Training Starts...")

        gb = GradientBoostingClassifier()

        grid_search_gb = GridSearchCV(gb, gb_params, cv=5)

        grid_search_gb.fit(X_train, y_train)

        print(grid_search_gb.best_params_)

        print("GradientBoost Classifier Training Ends...")

    

    if 'svm' in classifiers:

        #SVM Classifier

        print("SVM Classifier Training Starts...")

        svm = SVC(probability=True)

        grid_search_svm = GridSearchCV(svm, svc_params, cv=5)

        grid_search_svm.fit(X_train, y_train)

        print(grid_search_svm.best_params_)

        print("SVM Classifier Training Ends...")

    

    if full_dataset == 'Y':

        print("Logistic Regression Accuracy on Traning Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_train),y_train))

        print("Random Forest Accuracy on Traning Data ",

                  accuracy_score(grid_search_rf.best_estimator_.predict(X_train),y_train))

        print("ExtraTree Accuracy on Traning Data ",

                  accuracy_score(grid_search_et.best_estimator_.predict(X_train),y_train))

        print("AdaBoost Accuracy on Traning Data ",

                  accuracy_score(grid_search_ada.best_estimator_.predict(X_train),y_train))

        print("GradientBoost Accuracy on Traning Data ",

                  accuracy_score(grid_search_gb.best_estimator_.predict(X_train),y_train))

        print("SVM Accuracy on Traning Data ",

                  accuracy_score(grid_search_svm.best_estimator_.predict(X_train),y_train))

        

        print("For Full dataset, 1 dataframe contains the train set and 1 contains the test set")    

        

        predictions = grid_search_lr.best_estimator_.predict(X_train)        

        train_stack_df['Logistic'] = predictions

        predictions = grid_search_lr.best_estimator_.predict(X_test)        

        test_stack_df['Logistic'] = predictions



        predictions = grid_search_rf.best_estimator_.predict(X_train)        

        train_stack_df['RandomForest'] = predictions

        predictions = grid_search_rf.best_estimator_.predict(X_test)        

        test_stack_df['RandomForest'] = predictions



        predictions = grid_search_et.best_estimator_.predict(X_train)        

        train_stack_df['ExtraTrees'] = predictions

        predictions = grid_search_et.best_estimator_.predict(X_test)        

        test_stack_df['ExtraTrees'] = predictions



        predictions = grid_search_ada.best_estimator_.predict(X_train)        

        train_stack_df['AdaBoost'] = predictions

        predictions = grid_search_ada.best_estimator_.predict(X_test)        

        test_stack_df['AdaBoost'] = predictions



        predictions = grid_search_gb.best_estimator_.predict(X_train)        

        train_stack_df['GradientBoost'] = predictions

        predictions = grid_search_gb.best_estimator_.predict(X_test)        

        test_stack_df['GradientBoost'] = predictions





        predictions = grid_search_svm.best_estimator_.predict(X_train)        

        train_stack_df['SVM'] = predictions

        predictions = grid_search_svm.best_estimator_.predict(X_test)        

        test_stack_df['SVM'] = predictions



    else:

        print("Logistic Regression Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

        print("Random Forest Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

        print("ExtraTree Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

        print("AdaBoost Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

        print("GradientBoost Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

        print("SVM Accuracy on Validation Data ",

                  accuracy_score(grid_search_lr.best_estimator_.predict(X_test),y_test))

            

        print("Dataframe contains the validation set predicitons")    

        predictions = grid_search_lr.best_estimator_.predict(X_test)        

        test_stack_df['Logistic'] = predictions



        predictions = grid_search_rf.best_estimator_.predict(X_test)        

        test_stack_df['RandomForest'] = predictions



        predictions = grid_search_et.best_estimator_.predict(X_test)        

        test_stack_df['ExtraTrees'] = predictions



        predictions = grid_search_ada.best_estimator_.predict(X_test)        

        test_stack_df['AdaBoost'] = predictions



        predictions = grid_search_gb.best_estimator_.predict(X_test)        

        test_stack_df['GradientBoost'] = predictions



        predictions = grid_search_svm.best_estimator_.predict(X_test)        

        test_stack_df['SVM'] = predictions

    

    return train_stack_df, test_stack_df
classifiers = ['lr','rf','ada','et','gb','svm']

train_stacking_df, test_stacking_df = classifiers_predict(X_train, y_train, X_test, y_test, 'Y', classifiers)

print(train_stacking_df.shape)

print(test_stacking_df.shape)
#train_stacking_df.to_csv("train_stacking_df.csv", index=False)

#test_stacking_df.to_csv("test_stacking_df.csv", index=False)
gbm_final = XGBClassifier(

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

 scale_pos_weight=1).fit(train_stacking_df, y_train)

predictions = gbm_final.predict(test_stacking_df)
# Generate Submission File 

def submission(predictions):

    Final_Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions})

    Final_Submission.to_csv("gender_submission.csv", index=False)
import keras

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, LeakyReLU, BatchNormalization

from keras.preprocessing.text import Tokenizer

from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD, Adam
# Building the model architecture with one layer of length 100

num_classes = 1

model = Sequential()

model.add(Dense(256, activation = 'relu', input_shape=(12,)))

model.add(Dropout(.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()



# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
def create_model():

    model = Sequential()

    model.add(Dense(256, activation = 'relu', input_shape=(12,)))

    model.add(Dropout(.5))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(.5))

    model.add(Dense(1, activation='sigmoid'))

    return model



def train():

   model = create_model()

   #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

   adam = Adam(lr=0.03, decay = 0.1)

   model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])



   checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

   model.fit(X_train, y_train, nb_epoch=50, batch_size=32, validation_split=0.2, verbose=2, callbacks=[checkpointer])



train()
def load_trained_model(weights_path):

    model = create_model()

    model.load_weights(weights_path)

    return model



model = load_trained_model("weights.hdf5")

predictions = model.predict_classes(X_test)

submission(predictions.ravel())
test_stacking_df_NN = test_stacking_df.copy()

model = load_trained_model("weights.hdf5")

test_stacking_df_NN['NN'] = model.predict_classes(X_test).ravel()

predictions = test_stacking_df_NN.mode(axis=1)

submission(predictions.values.ravel())