# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt # for plotting graphs

import seaborn as sns

from sklearn.model_selection import train_test_split # for splitting the dataset

from sklearn.metrics import mean_absolute_error      # for finding mean squared error



# ml algorithms

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor    # for stacking models

from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso



# preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures



# hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



# feature selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.ensemble import ExtraTreesClassifier

import os

print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
df_train.describe()
df_test.describe()
test_index=df_test['Unnamed: 0']
def checking_null(X):

    return X.isnull().values.any()

    

print(checking_null(df_train))

print(checking_null(df_test))
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)

continuous = ['F6','F10','F13','F14','F15','F16','F17']      

categorical = ['F3','F4','F5','F7','F8','F9','F11','F12']
 

def correlation(X,cols):

    corr = X[cols].corr()

    return corr.style.background_gradient(cmap='coolwarm')

# finding the correlation matrix of the dataset

col = df_train.columns

correlation(df_train,col)
def feature_engineering_train(X):

    X['F18'] = (X['F15'] + X['F13'])/2

    

    # rearranging the columns to the normal order

    cols = X.columns.tolist()

    cols = ['F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','O/P']

    X = X[cols]

    return X

def feature_engineering_test(X):

    X['F18'] = (X['F15'] + X['F13'])/2

   

    

    return X

    

df_train = feature_engineering_train(df_train)

df_test = feature_engineering_test(df_test)
X = df_train.loc[:, 'F3':'F18']

y = df_train.loc[:, 'O/P']

test_X = df_test.loc[:,'F3':'F18']


#applying SelectKBest class to extract best features

bestfeatures = SelectKBest(score_func=f_classif, k=15)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Feature_Importance']  #naming the dataframe columns

print(featureScores.nlargest(15,'Feature_Importance'))  #printing the features in the order of their score



featureScores.plot(kind = 'bar',x = 'Specs',y = 'Feature_Importance')

plt.show()
# finding the distribution of continuous variables 

for col in continuous:

    sns.distplot(X[col], hist_kws=dict(color='plum',    edgecolor="k", linewidth=1))

    plt.show()
# visualizing the categroical variables v/s the target variable

for feature in categorical:

    data=df_train.copy()

    data.groupby(feature)['O/P'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('O/P')

    plt.title(feature)

    plt.show()
def drop_columns(X):

    

    

    X.drop(['F6','F10','F14','F15','F13'], axis = 1, inplace = True)

    return X

drop_columns(X)

drop_columns(test_X)

def standardising(X,column):

    

    scaler = StandardScaler()

   

    X_continuous_std = pd.DataFrame(data=scaler.fit_transform(X[column]), columns=column)   # standardizing the 

                                                                                                    # continuous variables



    X = pd.merge(X_continuous_std, X[categorical+continuous - column], left_index=True, right_index=True)   # replacing numeric columns with

                                                                                        # standardized entries

    cols  = X.columns.tolist()



    cols = ['F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17']  # rearranging the columns 

    X = X[cols]                                                                                  # back to normal order

    

    return X
# since standardizing reduced the performance of the model, it is not done

"""

X = standardising(X,continuous)

test_X = standardising(test_X,continuous)"""
print(X)
def one_hot_encode(X_train,categorical,numerical):

    low_cardinality_cols = [col for col in categorical if X_train[col].nunique() < 10] # removing categorical variables

                                                                                       # with many unique values

    X_train = X_train[categorical + numerical ]

    OHE=OneHotEncoder(handle_unknown='ignore',sparse=False)



    cat_features_train=pd.DataFrame(OHE.fit_transform(X_train[low_cardinality_cols]))  # encoding the categorical variables

    

    cat_features_train.index=X_train.index

    

    print(cat_features_train.shape)



    num_train=X_train[numerical]                  

 



    X_train=pd.concat([num_train,cat_features_train],axis=1)                        

    

    return X_train

   

        
# since one-hot encode reduced the performance of the model, it was not used.

"""

X = one_hot_encode(X,categorical,continuous)

test_X = one_hot_encode(test_X,categorical,continuous)

X, test_X = X.align(test_X,join='inner',axis=1)

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,shuffle=True,random_state = 42)
print(test_X)

print(X_train)
def hyperparameter_tuning_random_forest(X_train,y_train):

    

    # checking with different possible values for each hyperparameter 

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

   

    max_features = ['auto', 'sqrt']

   

    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

    max_depth.append(None)

   

    min_samples_split = [2, 5, 10]

   

    min_samples_leaf = [1, 2, 4]

   

    bootstrap = [True, False]

    # Creating the random grid

    random_grid = {'n_estimators': n_estimators,

                   'max_features': max_features,

                   'max_depth': max_depth,

                   'min_samples_split': min_samples_split,

                   'min_samples_leaf': min_samples_leaf,

                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameter

    # First create the base model to tune

    rf = RandomForestRegressor()

    # Random search of parameters, using 3 fold cross validation, 

    # search across 25 different combinations, and use all available cores

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 25, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # Fit the random search model

    rf_random.fit(X_train, y_train)

    return rf_random
# the best values for hyperparameters were found by running this and those values have been used in the model. hence currently commented out



#best_parameters = hyperparameter_tuning_random_forest(X_train,y_train).best_params_

def hyperparameter_tuning_lgbm(X_train,y_train):

   # checking with different possible values for each hyperparameter

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 50)]

   

    num_leaves = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]

   

    max_depth = [int(x) for x in np.linspace(1, 50, num = 11)]

    max_depth.append(None)



    

    learning_rate = [0.01,0.05,0.25]

    # Create the random grid

    random_grid = {'n_estimators': n_estimators,

                   'num_leaves': num_leaves,

                   'max_depth': max_depth,

                   'learning_rate': learning_rate}



    # Use the random grid to search for best hyperparameters

    # First create the base model to tune

    lgbm = LGBMRegressor()

    # Random search of parameters, using 3 fold cross validation, 

    # search across 25 different combinations, and use all available cores

    lgbm_random = RandomizedSearchCV(estimator = lgbm, param_distributions = random_grid, n_iter = 25, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # Fit the random search model

    lgbm_random.fit(X_train, y_train)

    return lgbm_random

# the best values for hyperparameters were found by running this and those values have been used in the model. hence currently commented out



#best_parameters = hyperparameter_tuning_lgbm(X_train,y_train).best_params_

def hyperparamter_tuning_xgb(X_train,y_train):

    # checking with different possible values for each hyperparameter

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 50)]

   

    min_child_weight = [int(x) for x in np.linspace(start = 0, stop = 10, num = 1)]

  

    max_depth = [int(x) for x in np.linspace(1, 50, num = 11)]

    max_depth.append(None)



   

    learning_rate = [0.01,0.05,0.25]

    # Create the random grid

    random_grid = {'n_estimators': n_estimators,

                   'min_child_weight': min_child_weight,

                   'max_depth': max_depth,

                   'learning_rate': learning_rate}



    # Use the random grid to search for best hyperparameters

    # First create the base model to tune

    xgb = XGBRegressor()

    # Random search of parameters, using 3 fold cross validation, 

    # search across 50 different combinations, and use all available cores

    xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # Fit the random search model

    xgb_random.fit(X_train, y_train)

    return xgb_random
#best_parameters = hyperparameter_tuning_xgb(X_train,y_train).best_params_

lgbm = LGBMRegressor(learning_rate=0.05,n_estimators= 1000,max_depth = 6)

rf = RandomForestRegressor(n_estimators=50,max_depth=17, random_state=42)

xgb = XGBRegressor(n_estimators = 1000, learning_rate = 0.05,random_state = 42)
stack = StackingCVRegressor(regressors=( rf, lgbm, xgb),

                            meta_regressor=xgb, cv=12,

                            use_features_in_secondary=True,

                            store_train_meta_features=True,

                            shuffle=False,

                            random_state=42)
stack.fit(X_train.values, y_train.values)


pred = stack.predict(X_test.values)

mae = mean_absolute_error(pred,y_test)

print(mae)
pred = stack.predict(X_train.values)

mae = mean_absolute_error(pred,y_train)

print(mae)
pred = stack.predict(test_X.values)

print(pred)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output.csv', index=False)
