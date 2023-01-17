# TITANIC COMPETETION - CLASSIFYING DEATH/SURVIVE OF PASSENGER

# MAJORITY DECISION OF RANDOM FOREST, REGULARIZED LOGISTIC REGRESSION AND NEURAL NETWORK
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# IMPORT PACKAGES

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

import string

from sklearn.metrics import accuracy_score

import numpy as np 

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

import random

import re

from matplotlib.legend_handler import HandlerLine2D

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegressionCV

from sklearn.preprocessing import MinMaxScaler

from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

from sklearn.impute import KNNImputer

import numpy as np

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import PolynomialFeatures



# further presets

pd.set_option('display.expand_frame_repr', False)

random_state = 1000
# FIRST STEPS



# Read the data

train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')



# Exclude Columns not used for analysis

train = train.drop(['PassengerId'], axis = 1)

test = test.drop(['PassengerId'], axis = 1)



# Apply train set structrure to set set

test['Survived'] = 99

test = test.reindex(columns=list(train))



# Concetanete train and test to one dataframe

df_train_test = pd.concat([train, test], ignore_index=True, sort = False)



# Check

print('Number rows train:', train.shape[0])

print('Number rows test:', test.shape[0])

print('Number rows train + test:', df_train_test.shape[0])

print('Example view on data:')

print(train.iloc[50:55,:])

print( test.iloc[50:55,:])
# GENERAL PREPROCESSING - FEATURE ENGINEERING - 1



# NAME TO TITLE



list_titles = [ 'Lady', 'Countess', 'Dona','Capt', 'Col', 'Major','Rev','Jonkheer', 'Don','Sir','Mrs','Mme','Mlle','Ms', 'Master', 'Dr', 'Miss', 'Mr']



title_l = []

title_ll = []



for str in df_train_test.Name:

 str_1 = re.sub('[^A-Za-z0-9]+', ' ', str)

 str_2 = re.split(' ',str_1)

 str_3 = (list((set(str_2).intersection(list_titles))))

 str_4 = str_3[0]

 title_l.append(str_4)





for str in title_l:

    str = str.replace('Ms','Miss')

    str = str.replace('Mlle','Miss')

    str = str.replace('Mme', 'Mrs')

    str = str.replace('Lady','Royalty')

    str = str.replace('Countess','Royalty')

    str = str.replace('Dona','Royalty')

    str = str.replace('Capt','Officer')

    str = str.replace('Col','Officer')

    str = str.replace('Major','Officer')

    str = str.replace('Rev','Officer')

    str = str.replace('Jonkheer','Royalty')

    str = str.replace('Don','Royalty')

    str = str.replace('Sir','Royalty') 

    title_ll.append(str)



title = pd.Series(title_ll) 



# add to dataframe

df_train_test['Title'] = title

df_train_test = df_train_test.drop('Name', axis = 1)



# Check

print(df_train_test.head(10))
# GENERAL PREPROCESSING - FEATURE ENGINEERING - FILL NAS 1 



# 1. make own category of missing  cabins



df_train_test['Cab_missin']  = df_train_test.Cabin.isna().astype(int)  





# 2. derive some missing values from tickets



print(df_train_test[df_train_test.Ticket=='2668']) #example



df_na        = df_train_test.copy()

df_cab_na    = df_na[['Ticket','Cabin']]



# only rows where Cabin unequal Nan

df_cab = df_cab_na[df_cab_na.Cabin.notnull()]

df_cab = df_cab.sort_values(by=['Ticket'])

df_cab['nr'] = 1



# extract Cabin per Ticket

df_cab_ticket     = df_cab.groupby(['Ticket','Cabin']).count()

df_cab_ticket     = df_cab_ticket.add_suffix('_Count').reset_index()

df_cab_ticket_max = df_cab_ticket.groupby(['Ticket'])['nr_Count'].max()

idx = df_cab_ticket.groupby(['Ticket'])['nr_Count'].transform(max) == df_cab_ticket['nr_Count']

df  = df_cab_ticket[idx]

df  = df.drop(['nr_Count'], axis = 1)



df_shift = df['Ticket']

df_shift = df_shift.shift(-1)



df['Ticket_sh'] = df_shift



df.drop(df[df['Ticket'] == df['Ticket_sh']].index, inplace=True)

lookup = df.drop(['Ticket_sh'], axis = 1 )



print(lookup[lookup.Ticket=='2668'].head())



# Fill Nas

print('Cols with Nas (before fill nas):\n',df_train_test.Cabin.isna().sum())

print(df_train_test.head(10))



map_lookup = lookup.set_index('Ticket')['Cabin']





df_train_test['Cabin']  = df_train_test['Cabin'].fillna(df_train_test['Ticket'].map(map_lookup))



print(df_train_test[df_train_test.Ticket=='2668']) #example



df_train_test = df_train_test.drop(['Ticket'], axis = 1)



# Check

print('Cols with Nas (after fill nas):\n',df_train_test.Cabin.isna().sum())

print(df_train_test.head())

# GENERAL PREPROCESSING - FEATURE ENGINEERING - 2



# 1. CABIN TO ROOM NUMBER



df_train_test['Cabin']       =  df_train_test['Cabin'].fillna('Unknown') # first fill nas 





cabin_nr = []



for str in df_train_test.Cabin: #extract room number of Cabin column

    if str == 'Unknown': # if cabin and room number missing

     str = '99999'

    else:

     str = str[1:len(str)]

    str = str.split(" ")[0]

    if str == "": #if room number is mising

      str = "999"  

    cabin_nr.append(str)



cab_nr = pd.Series(cabin_nr) 



# add to dataframe

df_train_test['Cabin_Nr'] = cab_nr.astype('int32')





# 2. CABIN TO CABIN CATEGORY



cabin_l = []



for str in df_train_test.Cabin: #extract Cabin category of Cabin column

    cabin_l.append(str[0])



# add to dataframe

df_train_test['Cabin_cat'] = cabin_l

df_train_test = df_train_test.drop('Cabin', axis = 1)



# set again unknwon to Nan

df_train_test.loc[df_train_test['Cabin_cat'] == 'U', 'Cabin_cat'] = np.nan #fill later Nas



# Check

print(df_train_test.head(10))

# GENERAL PREPROCESSING - FEATURE ENGINEERING - FILL NAS 2



# Fill Nas

print('Cols with Nas (before fill nas):\n',df_train_test.isna().sum())



print(df_train_test.head(10))



# Multivariate Imputing

df_train_test['Age']  = df_train_test.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median(skipna=True))).astype(int)



# Simple Imputing because of very low proportion

df_train_test['Embarked'] =  df_train_test['Embarked'].fillna('S')

df_train_test['Fare']     =  df_train_test['Fare'].fillna(df_train_test['Fare'].median(skipna=True))



# Own Category

df_train_test['Cabin_cat'] =  df_train_test['Cabin_cat'].fillna('U')



# Check

print('Cols with Nas (after fill nas):\n',df_train_test.isna().any()[lambda x: x],'\n')

print(df_train_test.head(10))
#ONE HOT ENCODING (APPRACH FOR RANDOM FOREST - ADDITONAL DUMMIES FOR NN AND LR LATER)

df_train_test = pd.get_dummies(df_train_test)

print(df_train_test.head(1).T)
# CREATE TRAIN UND TEST SET -  RANDOM FOREST



# seperate Train and test of df_train_test

len_train = train.shape[0]

test_RF   = df_train_test.iloc[len_train:,]

train_RF  = df_train_test.iloc[:len_train,]



# check

print('RANDOM FOREST TRAIN TEST SPLIT\n')

print(train_RF.tail())

print(test_RF.head(),'\n')

 

# exclude 'Survived' of Test

test_RF = test_RF.drop('Survived', axis = 1)



# check

print('number rows Train:',  train_RF.shape[0], ' / number rows Test:', test_RF.shape[0])



# Train test split val

train_val_RF, test_val_RF = train_test_split(train_RF, test_size = 0.3, random_state = random_state)



print('Train set RF  Survived distribution: ', round(train_val_RF['Survived'].value_counts()/train_val_RF['Survived'].count(),1).T)

print('Test  set RF  Survived distribution: ',  round(test_val_RF['Survived'].value_counts()/test_val_RF['Survived'].count(),1))



# 1. exclude target variable from train set

y_train_val_RF = train_val_RF['Survived']

y_test_val_RF  =  test_val_RF['Survived']



# 2. exclude target variable from train set

X_train_val_RF = train_val_RF.drop(['Survived'], axis=1)

X_test_val_RF  = test_val_RF.drop(['Survived'], axis=1)
##### MODEL UND EVALUATION - RANDOM FOREST 



# GridSearch Tuningparamter

RF_grid = {

            'max_features': [18,20,22,25],

            'min_samples_leaf': [3,5,8],

            'min_samples_split': [2,3,5],

            'n_estimators': [280,300,330]

}

print('RANDOM FOREST\n\n','Grid-Setup Random Forest : ', RF_grid)



# Modellerstellung

RF = RandomForestClassifier( random_state = random_state)

RF_grid_search = GridSearchCV(estimator = RF, param_grid = RF_grid, scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 0)

RF_grid_search.fit(X_train_val_RF,y_train_val_RF)



# Evaluation on Test set

print('MAX Test Scores while validation:' , round(max(RF_grid_search.cv_results_['mean_test_score']),2)) 

print('MIN Test Scores while validation:' , round(min(RF_grid_search.cv_results_['mean_test_score']),2)) 

best_grid_RF = RF_grid_search.best_estimator_



y_pred_train_val_RF = best_grid_RF.predict(X_train_val_RF)

y_pred_test_val_RF  = best_grid_RF.predict(X_test_val_RF)



acc_train_RF = round(accuracy_score(y_train_val_RF, y_pred_train_val_RF),2)

acc_test_RF  = round(accuracy_score(y_test_val_RF,   y_pred_test_val_RF),2)



print('Accuracy-Score Train:', acc_train_RF,'Accuracy-Score Test:', acc_test_RF)

print('Best Parameter values:', RF_grid_search.best_params_) 
# FURTHER FEATURE ENGINEEERING - CREATE TRAIN UND TEST SET -  REGULARIZED LOGISITC REGRESSION



# SPECIAL PREPROCESSING - FEATURE ENGINEERING - BINNING - LR



df_train_test_LR = df_train_test.copy()



# Feature Binning Fare

bins_fare  = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100000] 

labels_fare = ["Fare_(-1,0]","Fare_(0,10]","Fare_(10,20]","Fare_(20,30]","Fare_(30,40]","Fare_(40,50]","Fare_(50,60]","Fare_(60,70]","Fare_(70,80]","Fare_(80,90]","Fare_(90,1000]"]

df_train_test_LR['Fare']  = pd.cut(df_train_test_LR['Fare'], bins = bins_fare, labels = labels_fare)



# Feature Binning Age

bins_age  = [-1, 0, 1, 5, 15, 20, 30, 50, 65, 100] 

labels_age = ["Age_(-1,0]","Age_(0,1]","Age_(1,5]","Age_(5,15]","Age_(15,20]","Age_(20,30]","Age_(30,50]","Age_(50,65]","Age_(65,100]"]

df_train_test_LR['Age'] = pd.cut(df_train_test_LR['Age'], bins = bins_age, labels = labels_age)



# Feature Binning Cabin_Nr

bins_cab  = [0, 20, 40, 60, 80, 100, 120, 150,  1000, 100000] 

labels_cab = ["cab_(0,20]","cab_(20,40]","cab_(40,60]","cab_(60,80]","cab_(80,100]","cab_(100,120]","cab_(120,150]","cab_999","cab_99999"]

df_train_test_LR['Cabin_Nr'] = pd.cut(df_train_test_LR['Cabin_Nr'], bins = bins_cab, labels = labels_cab)



# One Hot Encoding Binned Age

X_Age            = pd.get_dummies(df_train_test_LR['Age'])

df_train_test_LR = df_train_test_LR.drop('Age', axis = 1)

df_train_test_LR = pd.concat([df_train_test_LR, X_Age], axis = 1)



# One Hot Encoding Binned Fare

X_Fare           = pd.get_dummies(df_train_test_LR['Fare'])

df_train_test_LR = df_train_test_LR.drop('Fare', axis = 1)

df_train_test_LR = pd.concat([df_train_test_LR, X_Fare], axis = 1)



# One Hot Encoding Binned Cab Nr

X_Cabin_Nr       = pd.get_dummies(df_train_test_LR['Cabin_Nr'])

df_train_test_LR = df_train_test_LR.drop('Cabin_Nr', axis = 1)

df_train_test_LR = pd.concat([df_train_test_LR, X_Cabin_Nr], axis = 1)





# exclude target variable for adding ineractions

y_df_train_test_LR = df_train_test_LR['Survived']

df_train_test_LR   = df_train_test_LR.drop(['Survived'], axis=1)



# Add interactions

interaction            =  PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)

df_train_test_LR_inter =  interaction.fit_transform(df_train_test_LR)

df_train_test_LR_inter =  pd.DataFrame(df_train_test_LR_inter)



# minmax Scaling for LR

scaler = MinMaxScaler(feature_range = (0,1))

df_train_test_LR_inter_scaled         =  scaler.fit_transform(df_train_test_LR_inter)

df_train_test_LR_inter_scaled         =  pd.DataFrame(df_train_test_LR_inter_scaled)



# seperate Train and test of df_train_test

df_train_test_LR = pd.concat([df_train_test_LR_inter_scaled, y_df_train_test_LR], axis = 1)

len_train = train.shape[0]

test_LR   = df_train_test_LR.iloc[len_train:,]

train_LR  = df_train_test_LR.iloc[:len_train,]

test_LR   = test_LR.drop(['Survived'], axis = 1)



print('REGULARIZED LOGISITC REGRESSION TRAIN TEST SPLIT\n')

 

# check

print('number rows Train:',  train_LR.shape[0], ' / number rows Test:', test_LR.shape[0])



# Train test split val

train_val_LR, test_val_LR = train_test_split(train_LR, test_size = 0.3, random_state = random_state)



print('Train set LR  Survived distribution: ', round(train_val_LR['Survived'].value_counts()/train_val_LR['Survived'].count(),1).T)

print('Test  set LR  Survived distribution: ',  round(test_val_LR['Survived'].value_counts()/test_val_LR['Survived'].count(),1))



# 1. exclude target variable from train set

y_train_val_LR = train_val_LR['Survived']

y_test_val_LR  =  test_val_LR['Survived']



# 2. exclude target variable from train set

X_train_val_LR = train_val_LR.drop(['Survived'], axis=1)

X_test_val_LR  = test_val_LR.drop(['Survived'], axis=1)
##### MODEL UND EVALUATION - LOGISTIC REGRESSION





# MODELING LR

# GridSearch Tuningparamter

LR_grid = {

           'penalty' : ['l1', 'l2'],

           'C' : np.logspace(-4, 6, 100),

}

#print('Grid-Setup : ', lr_grid)





LR = LogisticRegression(max_iter = 1000, solver = 'liblinear')

LR_grid_search = GridSearchCV(estimator = LR, param_grid = LR_grid, scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 0)

LR_grid_search.fit(X_train_val_LR, y_train_val_LR)



# Evaluation

print('MAX Test Scores while validation:' , round(max(LR_grid_search.cv_results_['mean_test_score']),2)) 

print('MIN Test Scores while validation:' , round(min(LR_grid_search.cv_results_['mean_test_score']),2)) 

best_grid_LR = LR_grid_search.best_estimator_



y_pred_train_val_LR = best_grid_LR.predict(X_train_val_LR)

y_pred_test_val_LR  = best_grid_LR.predict(X_test_val_LR)



acc_train_LR = round(accuracy_score(y_train_val_LR, y_pred_train_val_LR),2)

acc_test_LR  = round(accuracy_score(y_test_val_LR,   y_pred_test_val_LR),2)



print('Accuracy-Score on Train:', acc_train_LR,'Accuracy-Score on Test:', acc_test_LR)

print('Best parameter values:', LR_grid_search.best_params_)
# FURTHER FEATURE ENGINEEERING - CREATE TRAIN UND TEST SET -  NEURAL NETWORK



# SPECIAL PREPROCESSING - FEATURE ENGINEERING - BINNING - NN



df_train_test_NN = df_train_test.copy()



# Feature Binning Fare

bins_fare  = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100000] 

labels_fare = ["Fare_(-1,0]","Fare_(0,10]","Fare_(10,20]","Fare_(20,30]","Fare_(30,40]","Fare_(40,50]","Fare_(50,60]","Fare_(60,70]","Fare_(70,80]","Fare_(80,90]","Fare_(90,1000]"]

df_train_test_NN['Fare']  = pd.cut(df_train_test_NN['Fare'], bins = bins_fare, labels = labels_fare)



# Feature Binning Age

bins_age  = [-1, 0, 1, 5, 15, 20, 30, 50, 65, 100] 

labels_age = ["Age_(-1,0]","Age_(0,1]","Age_(1,5]","Age_(5,15]","Age_(15,20]","Age_(20,30]","Age_(30,50]","Age_(50,65]","Age_(65,100]"]

df_train_test_NN['Age'] = pd.cut(df_train_test_NN['Age'], bins = bins_age, labels = labels_age)



# Feature Binning Cabin_Nr

bins_cab  = [0, 20, 40, 60, 80, 100, 120, 150,  1000, 100000] 

labels_cab = ["cab_(0,20]","cab_(20,40]","cab_(40,60]","cab_(60,80]","cab_(80,100]","cab_(100,120]","cab_(120,150]","cab_999","cab_99999"]

df_train_test_NN['Cabin_Nr'] = pd.cut(df_train_test_NN['Cabin_Nr'], bins = bins_cab, labels = labels_cab)



# One Hot Encoding Binned Age

X_Age            = pd.get_dummies(df_train_test_NN['Age'])

df_train_test_NN = df_train_test_NN.drop('Age', axis = 1)

df_train_test_NN = pd.concat([df_train_test_NN, X_Age], axis = 1)



# One Hot Encoding Binned Fare

X_Fare           = pd.get_dummies(df_train_test_NN['Fare'])

df_train_test_NN = df_train_test_NN.drop('Fare', axis = 1)

df_train_test_NN = pd.concat([df_train_test_NN, X_Fare], axis = 1)



# One Hot Encoding Binned Cab Nr

X_Cabin_Nr       = pd.get_dummies(df_train_test_NN['Cabin_Nr'])

df_train_test_NN = df_train_test_NN.drop('Cabin_Nr', axis = 1)

df_train_test_NN = pd.concat([df_train_test_NN, X_Cabin_Nr], axis = 1)



# exclude target variable for scaling

y_df_train_test_NN = df_train_test_NN['Survived']

df_train_test_NN   = df_train_test_NN.drop(['Survived'], axis=1)



# minmax Scaling for NN

scaler = MinMaxScaler(feature_range = (0,1))

df_train_test_NN_scaled         =  scaler.fit_transform(df_train_test_NN)

df_train_test_NN                =  pd.DataFrame(df_train_test_NN_scaled)



# seperate Train and test of df_train_test

df_train_test_NN = pd.concat([df_train_test_NN, y_df_train_test_NN], axis = 1)

len_train = train.shape[0]

test_NN   = df_train_test_NN.iloc[len_train:,]

train_NN  = df_train_test_NN.iloc[:len_train,]

test_NN   = test_NN.drop(['Survived'], axis = 1)



print('NEURAL NETWORK TRAIN TEST SPLIT\n')

 

# check

print('number rows Train:',  train_NN.shape[0], ' / number rows Test:', test_NN.shape[0])



# Train test split val

train_val_NN, test_val_NN = train_test_split(train_NN, test_size = 0.3, random_state = random_state)



print('Train set NN  Survived distribution: ', round(train_val_NN['Survived'].value_counts()/train_val_NN['Survived'].count(),1).T)

print('Test  set NN  Survived distribution: ',  round(test_val_NN['Survived'].value_counts()/test_val_NN['Survived'].count(),1))



# 1. exclude target variable from train set

y_train_val_NN = train_val_NN['Survived']

y_test_val_NN  =  test_val_NN['Survived']



# 2. exclude target variable from train set

X_train_val_NN = train_val_NN.drop(['Survived'], axis=1)

X_test_val_NN  = test_val_NN.drop(['Survived'], axis=1)
##### MODELL UND EVALUIERUNG - NEURONAL NETWORK





input_dim   = X_train_val_NN.shape[1]

print('Numer Input nodes:', input_dim)



# MODELING NN

NN = Sequential()

NN.add(Dense(input_dim, input_dim = input_dim, activation = 'softmax'))

NN.add(Dense(25, activation ='softmax'))

NN.add(Dense(1, activation ='sigmoid'))



NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

NN.fit(X_train_val_NN, y_train_val_NN, epochs=150, batch_size=50, verbose=0)

# _, accuracy_train = nn.evaluate(X_train_val_nn, y_train_val)

# _, accuracy_test  = nn.evaluate(X_test_val_nn, y_test_val)

# print(accuracy_train)

# print(accuracy_test)



# Evaluation

y_pred_train_val_NN = np.around(NN.predict(X_train_val_NN).flatten())

y_pred_test_val_NN  = np.around(NN.predict(X_test_val_NN).flatten())



acc_train_NN = round(accuracy_score(y_train_val_NN, y_pred_train_val_NN),2)

acc_test_NN  = round(accuracy_score(y_test_val_NN,   y_pred_test_val_NN),2)



print('Accuracy-Score on Train:', acc_train_NN,'Accuracy-Score on Test:', acc_test_NN)

# STACKED MODEL



random_add = random_state +  np.around(np.random.randint(200))



# WEAK LEARNER MODELL

# Erzeuge Datensatz  [Modell_RF,Modell_LR, Modell_NN, True]

y_pred_stack_RF = best_grid_RF.predict_proba(X_test_val_RF)[:,1]

y_pred_stack_LR = best_grid_LR.predict_proba(X_test_val_LR)[:,1]

y_pred_stack_NN = NN.predict(X_test_val_NN).flatten()

stack           = pd.DataFrame({'Modell_RF': y_pred_stack_RF, 'Modell_LR': y_pred_stack_LR,  'Modell_NN': y_pred_stack_NN, 'Survived': y_test_val_LR}) # y_test_val_LR=y_test_val_NN..



# Erzeuge Trainset und Testset auf Stack

train_stack, test_stack = train_test_split(stack , test_size = 0.5, random_state=random_state)

y_train_stack = train_stack['Survived']

y_test_stack  = test_stack['Survived']

X_train_stack = train_stack.drop(['Survived'], axis = 1)

X_test_stack  = test_stack.drop(['Survived'],  axis = 1)



# Einfaches Baummodell

tr = RandomForestClassifier(random_state=random_state, max_features= 1,min_samples_leaf=1, min_samples_split= 2,n_estimators= 1)

tr.fit(X_train_stack, y_train_stack)



y_pred_train_stack = tr.predict(X_train_stack)

y_pred_test_stack  = tr.predict(X_test_stack)

acc_train_stack    = round(accuracy_score(y_train_stack,  y_pred_train_stack),3)

acc_test_stack     = round(accuracy_score(y_test_stack,   y_pred_test_stack),3)

acc_train_nn       = round(accuracy_score(y_train_stack, np.around(X_train_stack.Modell_NN)),3)

acc_test_nn        = round(accuracy_score(y_test_stack,  np.around(X_test_stack.Modell_NN)),3)

acc_train_rf       = round(accuracy_score(y_train_stack,  np.around(X_train_stack.Modell_RF)),3)

acc_test_rf        = round(accuracy_score(y_test_stack,   np.around(X_test_stack.Modell_RF)),3)

acc_train_lr       = round(accuracy_score(y_train_stack, np.around(X_train_stack.Modell_LR)),3)

acc_test_lr        = round(accuracy_score(y_test_stack,  np.around(X_test_stack.Modell_LR)),3)



print('Stack and Majority Predictions:')

print('Accuracy Score Train Stack:', acc_train_stack, 'Accuracy Score Test Stack: ',acc_test_stack)



#########



# MAJORITY DECISION



# combine single predicted probabilities of the models

y_pred_maj_rf = best_grid_RF.predict_proba(X_test_val_RF)[:,1]

y_pred_maj_lr = best_grid_LR.predict_proba(X_test_val_LR)[:,1]

y_pred_maj_nn = NN.predict(X_test_val_NN).flatten()

maj = pd.DataFrame({'Modell_RF': y_pred_maj_rf , 'Modell_LR': y_pred_maj_lr, 'Modell_NN': y_pred_maj_nn, 'Survived': y_test_val_RF})



# create Train set and Testset on Stack

train_maj, test_maj = train_test_split(maj , test_size = 0.5, random_state=random_state)

y_train_maj = train_maj['Survived']

y_test_maj  = test_maj['Survived']

X_train_maj = train_maj.drop(['Survived'], axis = 1)

X_test_maj  = test_maj.drop(['Survived'],  axis = 1)



X_train_maj_sum = np.sum(X_train_maj, axis = 1)

X_test_maj_sum  = np.sum(X_test_maj, axis = 1)

X_train_maj_max = np.max(X_train_maj, axis = 1)

X_test_maj_max  = np.max(X_test_maj, axis = 1)





threshhold = 3* 0.5

y_pred_train_maj = (X_train_maj_sum > threshhold).astype(int)

y_pred_test_maj  = (X_test_maj_sum  > threshhold).astype(int)





acc_train_maj   = round(accuracy_score(y_train_maj, y_pred_train_maj),3)

acc_test_maj    = round(accuracy_score(y_test_maj, y_pred_test_maj),3)



print('Accuracy Score Train Maj:  ', acc_train_maj,   'Accuracy Score Test Maj:   ',acc_test_maj)



print('Compare to single predictions:')

print('Accuracy Score Train NN:   ', acc_train_nn,    'Accuracy Score Test NN:    ',acc_test_nn)

print('Accuracy Score Train RF:   ', acc_train_rf,    'Accuracy Score Test RF:    ',acc_test_rf)

print('Accuracy Score Train LR:   ', acc_train_lr,    'Accuracy Score Test LR:    ',acc_test_lr)
# PREPARE SUBMISSION





# SINGLE PREDICTION FOR COMPARISION - not used

y_prediction_rf = best_grid_RF.predict(test_RF)

y_prediction_lr = best_grid_LR.predict(test_LR)

y_prediction_nn = np.around(NN.predict(test_NN).flatten())

                    

                            

# MAJORITY DECISION

y_pred_rf_maj = best_grid_RF.predict_proba(test_RF)[:,1]

y_pred_lr_maj = best_grid_LR.predict_proba(test_LR)[:,1]

y_pred_nn_maj = NN.predict(test_NN).flatten()



Y_maj = pd.DataFrame({'Modell_RF': y_pred_rf_maj , 'Modell_LR': y_pred_lr_maj, 'Modell_NN': y_pred_nn_maj})

Y_maj_sum = np.sum(Y_maj, axis = 1)

y_prediction_maj =  (Y_maj_sum > threshhold).astype(int)



##############



# OUTPUT 

Survived_sub = y_prediction_maj

# CREATE SUBMISSION

pass_id       = pd.read_csv('../input/titanic/test.csv')

my_submission = pd.DataFrame({'PassengerId': pass_id.PassengerId, 'Survived': Survived_sub})

my_submission.to_csv('submission.csv', index = False)

print(my_submission.head())