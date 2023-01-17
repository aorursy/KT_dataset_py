# Common libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Restrict minor warnings

import warnings

warnings.filterwarnings('ignore')
# Import test and train data

df_train = pd.read_csv('../input/train.csv')

df_Test = pd.read_csv('../input/test.csv')

df_test = df_Test
# From both train and test data

df_train = df_train.drop(['Soil_Type7', 'Soil_Type15','Soil_Type8', 'Soil_Type25'], axis = 1)

df_test = df_test.drop(['Soil_Type7', 'Soil_Type15','Soil_Type8', 'Soil_Type25'], axis = 1)



# Also drop 'Id'

df_train = df_train.iloc[:,1:]

df_test = df_test.iloc[:,1:]
# Taking only non-categorical values

Size = 10

X_temp = df_train.iloc[:,:Size]

X_test_temp = df_test.iloc[:,:Size]
r,c = df_train.shape

X_train = np.concatenate((X_temp,df_train.iloc[:,Size:c-1]),axis=1)

y_train = df_train.Cover_Type.values
r,c = df_test.shape

X_test = np.concatenate((X_test_temp, df_test.iloc[:,Size:c]), axis = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

# Setting parameters

x_data, x_test_data, y_data, y_test_data = train_test_split(X_train, y_train, test_size = 0.3)

rf_para = [{'n_estimators':[50, 100], 'max_depth':[5,10,15], 'max_features':[0.1, 0.3], \

           'min_samples_leaf':[1,3], 'bootstrap':[True, False]}]
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

rfc = GridSearchCV(RandomForestClassifier(), param_grid=rf_para, cv = 10, n_jobs=-1)

rfc.fit(x_data, y_data)

rfc.best_params_

rfc.grid_scores_
print ('Best accuracy obtained: {}'.format(rfc.best_score_))

print ('Parameters:')

for key, value in rfc.best_params_.items():

    print('\t{}:{}'.format(key,value))
# Best params: {'max_features': 0.3, 'n_estimators': 100, 'bootstrap': False, 'max_depth': 15, 'min_samples_leaf': 1}

RFC = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=0.3, bootstrap=True, min_samples_leaf=1,\

                             n_jobs=-1)

RFC.fit(X_train, y_train)

# y_pred = RFC.predict(X_test)
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

from sklearn.cross_validation import StratifiedKFold

from scipy.stats import randint, uniform
cv = StratifiedKFold(y_train, n_folds=10, shuffle=True)



params_dist_grid = {

    'max_depth': [1, 5, 10],

    'gamma': [0, 0.5, 1],

    'n_estimators': randint(1, 1001), # uniform discrete random distribution

    'learning_rate': uniform(), # gaussian distribution

    'subsample': uniform(), # gaussian distribution

    'colsample_bytree': uniform(), # gaussian distribution

    'reg_lambda':uniform(),

    'reg_alpha':uniform()

    }



xgbc_fixed = {'booster':['gbtree'], 'silent':1}
bst_gridd = RandomizedSearchCV(estimator=XGBClassifier(*xgbc_fixed), param_distributions=params_dist_grid,\

                               scoring='accuracy', cv=cv, n_jobs=-1)
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

# Calculating error

XGBC.fit(x_data, y_data, early_stopping_rounds=100, eval_set=eval_set, eval_metric='merror', verbose=True)



pred = XGBC.predict(x_test_data)



accuracy = accuracy_score(y_test_data, pred);

print ('accuracy:%0.2f%%'%(accuracy*100))
xgbc_pred= XGBC.predict(X_test)
# saving to a csv file to make submission

solution = pd.DataFrame({'Id':df_Test.Id, 'Cover_Type':xgbc_pred}, columns = ['Id','Cover_Type'])

solution.to_csv('Xgboost_sol.csv', index=False)