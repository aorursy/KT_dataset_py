# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

df_Test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
#drop soil types which has values only zero(soil_Type7 and soil_Type15)

#drop soil types which has only singe one(soil_Type8 and soil_Type25)

#from train and test data

#Data Frame is 2 dimensional object with two axes. Axis =0 and axis =1. Axis = 0 represents row and

#axis = a represents column.

df_train = df_train.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25'], axis =1)

#Moving values to a temporary test variable form Test variable

df_test = df_Test.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25'], axis =1)
df_train.dtypes

#Taking only non-categorical values

Size = 10 

X_temp = df_train.iloc[:, :Size]

X_test_temp = df_test.iloc[:, :Size]

X_temp
#Doubt: But Horizontal_distance to fire point is not a categorical variable.

r, c = df_train.shape

df_train.iloc[:,Size:c-1]
#Doubt: why to split and then concanenate to get the same set

r,c = df_train.shape

X_train = np.concatenate((X_temp,df_train.iloc[:,Size:c-1]),axis=1)

y_train = df_train.Cover_Type.values
r,c = df_test.shape

X_test = np.concatenate((X_test_temp, df_test.iloc[:,Size:c]), axis = 1)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

X_data, x_test_data, y_data, y_test_data = train_test_split(X_train, y_train, test_size = 0.3)

#n_estimator: number of trees , can consider 50 and 100 and choose the best,

#

#min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered 

#if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of 

#smoothing the model, especially in regression.

#

#Bootstrap : The idea is to repeatedly sample data with replacement from the original training set

#in order to produce multiple separate training sets. These are then used to allow "meta-learner" or "ensemble" methods

#to reduce the variance of their predictions,thus greatly improving their predictive performance.

#

rf_para = [{'n_estimators':[50, 100], 'max_depth':[5, 10, 15], 'max_features':[0.1, 0.3],\

            'min_samples_leaf':[1,3], 'bootstrap':[True, False]}]
rfc = GridSearchCV(RandomForestClassifier(), param_grid = rf_para, cv = 10, n_jobs = -1)

rfc.fit(X_data, y_data)



rfc.best_params_
#rfc.grid_scores_
#RFC = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=0.3, bootstrap=True, min_samples_leaf=1,\

#                             n_jobs=-1)

#kaggle score 0.60

#when the below parameters were added, in the classification report the accuracy score hiked from 0.83 to 0.97

#kaggle score 0.70400

RFC = RandomForestClassifier(n_estimators=100, max_depth=15, max_features=0.3, bootstrap='False', min_samples_leaf=1,\

                             n_jobs=-1)

RFC.fit(X_train, y_train)
from sklearn.metrics import classification_report

Y_val_pred = RFC.predict(x_test_data)

target = ['class1', 'class2','class3','class4','class5','class6','class7' ]

print (classification_report(y_test_data, Y_val_pred, target_names=target))
Y_pred = RFC.predict(X_test)
solution = pd.DataFrame({'Id':df_Test.Id, 'Cover_Type':Y_pred}, columns = ['Id','Cover_Type'])

solution.to_csv('RFCcover_sol.csv', index=False)
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(model,title, X, y,n_jobs = 1, ylim = None, cv = None,train_sizes = np.linspace(0.1, 1, 5)):

    

    # Figrue parameters

    plt.figure(figsize=(10,8))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel('Training Examples')

    plt.ylabel('Score')

    

    train_sizes, train_score, test_score = learning_curve(model, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    # Calculate mean and std

    train_score_mean = np.mean(train_score, axis=1)

    train_score_std = np.std(train_score, axis=1)

    test_score_mean = np.mean(test_score, axis=1)

    test_score_std = np.std(test_score, axis=1)

    

    plt.grid()

    plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std,\

                      alpha = 0.1, color = 'r')

    plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std,\

                    alpha = 0.1, color = 'g')

    

    plt.plot(train_sizes, train_score_mean, 'o-', color="r", label="Training score")

    plt.plot(train_sizes, test_score_mean, 'o-', color="g", label="Cross-validation score")

    

    plt.legend(loc = "best")

    return plt
# Plotting Learning Curve

title = 'Learning Curve(Random Forest)'

model = RFC

cv = ShuffleSplit(n_splits=50, test_size=0.2,random_state=0)

plot_learning_curve(model,title,X_train, y_train, n_jobs=-1,ylim=None,cv=cv)

plt.show()
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from scipy.stats import randint,uniform 
cv = StratifiedKFold(y_train, n_folds = 10, shuffle = True)



params_dist_grid = {

    'max_depth':[1, 5, 10],

    'gamma':[0, 0.5,1],

    'n_estimators': randint(1, 1001),

    'learning_rate':uniform(),

    'subsample':uniform(),

    'colsample_bytree': uniform(),

    'reg_lambda': uniform(),

    'reg_alpha': uniform()

}



xgbc_fixed = {'booster':['gbtree'], 'silent':1}
best_gridd = RandomizedSearchCV(estimator = XGBClassifier(*xgbc_fixed), param_distributions  = params_dist_grid\

                               ,scoring = 'accuracy', cv = cv, n_jobs = -1)
# bst_gridd.fit(X_train, y_train)

# bst_gridd.grid_scores_



# print ('Best accuracy obtained: {}'.format(bst_gridd.best_score_))

# print ('Parameters:')

# for key, value in bst_gridd.best_params_.items():

    # print('\t{}:{}'.format(key,value))
# Best parameters selected using code in above cell

# Splitting the train data to test the best parameters

from sklearn.model_selection import train_test_split

seed = 123

x_data, x_test_data, y_data, y_test_data = train_test_split(X_train, y_train, test_size = 0.3,random_state=seed)



eval_set = [(x_test_data, y_test_data)]



XGBC = XGBClassifier(silent=1,n_estimators=641,learning_rate=0.2,max_depth=10,gamma=0.5,nthread=-1,\

                    reg_alpha = 0.05, reg_lambda= 0.35, max_delta_step = 1, subsample = 0.83, colsample_bytree = 0.6)
XGBC.fit(x_data, y_data, early_stopping_rounds=100, eval_set=eval_set, eval_metric='merror', verbose=True)



pred = XGBC.predict(x_test_data)



accuracy = accuracy_score(y_test_data, pred);

print ('accuracy:%0.2f%%'%(accuracy*100))
XGBC.fit(X_train, y_train)

xgbc_pred= XGBC.predict(X_test)
# saving to a csv file to make submission

solution = pd.DataFrame({'Id':df_Test.Id, 'Cover_Type':xgbc_pred}, columns = ['Id','Cover_Type'])

solution.to_csv('Xgboost_sol.csv', index=False)