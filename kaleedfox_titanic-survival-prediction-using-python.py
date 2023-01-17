# Importing required modules

import glob

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

sns.set_style("whitegrid")

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10,10 )
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.shape
# Highlight all females with the their survival column and assign them to a variable "female"

female = train_data.loc[train_data.Sex == "female"]["Survived"]

print("% of the women survived: ", (sum(female)/len(female))*100)
male = train_data.iloc[(train_data.Sex=="male").values, [1]]

print("% of the men survived: ", (sum(male.values)/len(male.values))*100)
#check for missing values

train_data.isnull().sum()
# Percntage of missing values:

missing_perc = pd.DataFrame((train_data.isnull().sum()/len(train_data) *100), columns=["% of missing values"])

missing_perc.sort_values(by = ["% of missing values"], ascending = False)
train_data.shape
# Age details

train_data.Age.describe()
train_data[(train_data.SibSp > 0) & (train_data.Age.isnull())]
train_data[train_data.Name.str.startswith("Taylor") | train_data.Name.str.startswith("Sage") | train_data.Name.str.startswith("Caram")]
train_data.loc[(train_data.Age.isnull()) & (train_data.SibSp > 0), "Age"] = np.random.randint(40, 60)

#train_data.Age = np.where(((train_data.Age.isnull()) & (train_data.SibSp > 0)), np.random.randint(40, 60), train_data.Age)
train_data[(train_data.Parch > 0) & (train_data.Age.isnull())]
train_data[train_data.Name.str.startswith("Bourke")]
train_data.loc[((train_data.Age.isnull()) & (train_data.Name.str.startswith("Bourke"))), "Age"] = np.random.randint(2,12)
train_data.loc[(train_data.Parch > 0) & (train_data.Age.isnull()), "Age"] = np.random.randint(30,50)

train_data.Age =np.where(((train_data.Parch > 0) & (train_data.Age.isnull())), np.random.randint(30,50), train_data.Age) 
train_data.Fare.describe()
train_data[(train_data.Fare > 30) & (train_data.Age.isnull())]
train_data.loc[((train_data.Age.isnull())&(train_data.Fare > 30)), "Age"] = np.random.randint(50,80)
train_data.Age.describe()
train_data.Age.fillna(train_data.Age.mean(),inplace=True)
train_data.Age.dtype
train_data.head()
train_data.drop(["Cabin"], axis=1, inplace=True)
#embarked = pd.get_dummies(train_data.Embarked)

train_data["C"] = [1 if (x == "C") else 0 for x in train_data.Embarked]

train_data["S"] = [1 if (x  == "S") else 0 for x in train_data.Embarked]

train_data["Q"] = [1 if (x == "Q") else 0 for x in train_data.Embarked]

train_data["Male"] = [1 if (x == "male") else 0 for x in train_data.Sex]

train_data["Female"] = [1 if (x=="female") else 0 for x in train_data.Sex]



train_data = pd.concat([train_data, train_data[["C","S","Q"]]], axis=1)

train_data = pd.concat([train_data, train_data[["Male","Female"]]], axis=1)

train_data.drop(["Embarked", "Sex"], axis =1, inplace = True)
train_data.Ticket.describe()
train_data.drop(["Ticket"], axis=1, inplace=True)
train_data = train_data.loc[:,~train_data.columns.duplicated()]
train_data.corr()
plt.figure(figsize = (17,12))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(train_data.corr(),cmap=colormap, annot=True, fmt=".2f")
train_data.describe()
train_data.hist(figsize=(17,12))
train_data.columns
train_data.drop(["PassengerId", "Name","C","S","Q"], axis = 1,inplace=True)
train_data.columns
y = train_data.Survived
x = train_data.drop(["Survived"] ,axis = 1)
x.columns
# Split x,y to train, test with startified that provided with sklearn "train_test_split" with 80 training and 20 to test my model

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, ) # 75% training and 25% test
import lightgbm as lgb

#------------------------Build LightGBM Model-----------------------

trained=lgb.Dataset(x_train, label=y_train)



#Select Hyper-Parameters

params = {'boosting_type': 'gbdt',

          'objective': 'binary',

          'nthread': 5,

          'learning_rate': 0.05,

          'subsample_for_bin': 200,

          'subsample_freq': 1,

          'colsample_bytree': 0.8,

          'reg_alpha': 1.2,

          'reg_lambda': 1.2,

          'min_split_gain': 0.5,

          'min_child_weight': .5,

          'num_class' : 1,

          'metric' : 'binary_logloss'

          }
gridParams = {

    'num_iterations':[100,200 ,400],

    'min_child_samples':range(1,10,80), 

    'min_child_weight': [0.3,0.5, 2],

    'n_estimators': [6,16,100,150],

    'num_leaves': [8,15, 64, 256],

    'random_state' : [42], 

    'subsample' : [0.7,1],

    'max_depth' : [-1,7,9,15],

    'scale_pos_weight':[0.1,0.3,0.8]



    }



# Create classifier to use

mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',

          objective = 'binary',

          n_jobs = 5, 

          silent = True,

          colsample_bytree = params['colsample_bytree'],               

          subsample_for_bin = params['subsample_for_bin'],

          nthread = params['nthread'],

          reg_alpha = params['reg_alpha'],               

          reg_lambda = params['reg_lambda'],

          subsample_freq = params['subsample_freq'],

          min_split_gain = params['min_split_gain'],

          min_child_weight = params['min_child_weight'],

          metric = params['metric'])
 #Create the grid

from sklearn.model_selection import StratifiedKFold

num_folds=5

kfold=StratifiedKFold(n_splits=num_folds)

grid = GridSearchCV(mdl, gridParams, verbose=2, cv=kfold, n_jobs=-1)



# Run the grid

grid.fit(x_train, y_train)



# Print the best parameters found

print(grid.best_params_)

print(grid.best_score_)
# Using parameters already set above, replace in the best from the grid search

"""

params['n_estimators'] =  6

#params['learning_rate'] = grid.best_params_['learning_rate']

params['min_child_samples'] =  1

params['max_depth'] =  -1

params['random_state'] =  501

params['num_leaves' ]=  15

params['num_iterations' ]= 100

params['min_child_weight']=  .3

params['subsample'] =  .7

params['scale_pos_weight']=  0.83

# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

#print('Fitting with params: ')

#print(params)

"""

params = {'max_depth': -1, 'min_child_samples': 1, 'min_child_weight': 2, 'n_estimators': 6,

          'num_iterations': 100, 'num_leaves': 15, 'random_state': 42, 'scale_pos_weight': 0.8, 'subsample': 0.7}

lgbm = lgb.train(params,

                 train_set=trained,

                 valid_sets = trained,

                 num_boost_round = 280,

                 #early_stopping_rounds= 40,

                 verbose_eval= 100

                 )



#Predict on test set

predictions_lgbm_prob = lgbm.predict(x_test)

predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

acc_lgbm = accuracy_score(y_test,predictions_lgbm_01)

acc_lgbm
#Train model on selected parameters and number of iterations

target_lables = ['Accept', 'Scrap']

lgbm = lgb.train(params,

                 train_set=trained,

                 valid_sets = trained,

                 num_boost_round = 280,

                 #early_stopping_rounds= 40,

                 verbose_eval= 100

                 )



#Predict on test set

predictions_lgbm_prob = lgbm.predict(x_test)

predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.3, 1, 0) #Turn probability to 0-1 binary output



#--------------------------Print accuracy measures and variable importances----------------------

#Plot Variable Importances

lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')



#--------------------------Print accuracy measures and variable importances----------------------

#Plot Variable Importances

#lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')



#print Classification Report

print(classification_report(y_test, predictions_lgbm_01, target_names = target_lables))





#Print accuracy

acc_lgbm = accuracy_score(y_test,predictions_lgbm_01)

print('Overall accuracy of Light GBM model:', acc_lgbm)



# Plot the confusion matrix

plt.figure()

cm = confusion_matrix(y_test, predictions_lgbm_01)

labels = ['Accepted', 'Scrap']

plt.figure(figsize=(8,6))

plt.rcParams.update({'font.size': 23})

sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

plt.show()
#Print Area Under Curve

plt.figure()

false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_lgbm_prob)

roc_auc = auc(false_positive_rate, recall)

plt.title('Receiver Operating Characteristic (ROC)')

plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.ylabel('Recall')

plt.xlabel('Fall-out (1-Specificity)')

plt.show()



print('AUC score:', roc_auc)
# get the kaggle test set and fit it in my model then submit the solution to kaggle site

# Convert test_data.Sex to binary and remove the columns that I did not use in my traing set

test_set = test_data.copy()

test_set["Male"] = [1 if (x == "male") else 0 for x in test_set.Sex]

test_set["Female"] = [1 if (x=="female") else 0 for x in test_set.Sex]

test_set.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked", "PassengerId"], axis =1, inplace = True)
predictions_lgbm_prob = lgbm.predict(test_set)

predictions = np.where(predictions_lgbm_prob > 0.3, 1, 0) #Turn probability to 0-1 binary output

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")