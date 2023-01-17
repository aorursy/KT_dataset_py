# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

test = test_data.copy()

y = data['Survived']

X = data.drop(['Cabin','Name','Survived','Ticket','PassengerId'],axis=1)

test.drop(['Cabin','Name','Ticket','PassengerId'],axis=1,inplace=True)

data.columns

#missing_values = (data.isnull().sum())

#print(missing_values[missing_values > 0])

#print(missing_values)

# Looking at the Data

data.head()
# Now Looking how the variables are in the Data

data.info()
data['Embarked'].value_counts()
data.describe()
# Plotting the Data to Form Insights

data.hist(bins=50,figsize=(20,15))

plt.show()
# Splitting the Data into Two Parts:

# Train and Test 

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
# Stratified Splitting based on Pclass

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index,test_index in split.split(data,data['Pclass']):

    strat_train_set = data.loc[train_index]

    strat_test_set = data.loc[test_index]
data['Pclass'].value_counts()/len(data)
strat_test_set['Pclass'].value_counts()/len(strat_test_set)
strat_train_set.head()
# Now copying Data to Perform Visualizations and see Correlations



data = strat_train_set.copy()

# Correlations

corr_matrix = data.corr()



corr_matrix['Survived'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes= ['Survived','Fare','Pclass','Age']

scatter_matrix(data[attributes],figsize=(15,10))
# What to do with missing values?



# Since the Attribute 'Cabin' has almost all the values Null, it would be better to drop it

# The Attribute 'Age' has some NUll values, so we will just replace it with the median value





# Since Embarked Has 2 missing values, I will just drop these two rows

strat_train_set.dropna(subset=['Embarked'],inplace=True)
# Before Data Cleaning, we have to separate the predictors and the labels



data = strat_train_set.drop("Survived",axis=1)



data_labels = strat_train_set["Survived"].copy()






# Making a Pipeline which will transform the whole Dataset

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



data_num = data.drop(["Name","Sex","Ticket","Cabin","Embarked","PassengerId"], axis=1)





num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler())

])



data_num_tr = num_pipeline.fit_transform(data_num)
data_num_tr
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



num_attribs = list(data_num)

cat_attribs = ["Name","Sex","Ticket","Cabin","Embarked"]



full_pipeline = ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("drop_columns","drop",["Ticket","Name","Cabin","PassengerId"]),

    ("OneHotSex",OneHotEncoder(),["Sex","Embarked"]),

])



data_prepared = full_pipeline.fit_transform(data)
data_prepared
# Let us use a Decision Tree Classifier on our Data

from sklearn.tree import DecisionTreeClassifier



tree_cls = DecisionTreeClassifier()

tree_cls.fit(data_prepared,data_labels)

# Now checking the mean squared error

from sklearn.metrics import mean_squared_error

data_predictions = tree_cls.predict(data_prepared)

tree_mse = mean_squared_error(data_labels,data_predictions)

tree_mse = np.sqrt(tree_mse)

tree_mse

# Let us now use Scikit-Learn's K-fold cross-validation feature

from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_cls, data_prepared, data_labels,

                       scoring = "neg_mean_squared_error", cv = 10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print('Scores :',scores)

    print('Mean : ',scores.mean())

    print('Standard Deviation : ',scores.std())

    

display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestClassifier

forest_cls = RandomForestClassifier()

forest_cls.fit(data_prepared,data_labels)



forest_predictions = forest_cls.predict(data_prepared)

forest_mse = mean_squared_error(forest_predictions, data_labels)

forest_mse = np.sqrt(forest_mse)



forest_mse
scores = cross_val_score(forest_cls, data_prepared, data_labels,

                       scoring = "neg_mean_squared_error", cv = 10)

forest_rmse_scores = np.sqrt(-scores)



display_scores(forest_rmse_scores)
# Now let us fine tune the model



from sklearn.model_selection import GridSearchCV



param_grid = [

        {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

        {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}

]

forest_cls = RandomForestClassifier()



grid_search = GridSearchCV(forest_cls, param_grid, cv=5,

                          scoring = 'neg_mean_squared_error',

                          return_train_score=True)

grid_search.fit(data_prepared,data_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

    print(np.sqrt(-mean_score),params)
# Final Model
data = pd.read_csv('../input/train.csv')
data.dropna(subset=['Embarked'],inplace=True)

data_labels = data['Survived']

data = data.drop('Survived',axis=1)
data_final = full_pipeline.fit_transform(data)
rand_forest_clf = RandomForestClassifier()

rand_forest_clf.fit(data_final,data_labels)
from sklearn.metrics import accuracy_score

accuracy_score(data_labels,rand_forest_clf.predict(data_final))
# # Hyperparameters Tuning of Reandom Forest

# # param_grid = [

# #     {'n_estimators':[10,30,50],'max_features':[6,8,10]},

# #     {'bootstrap':[False],'n_estimators':[10,30,50],'max_features':[4,6,8,10]}

# # ]



# param_grid = {

#     'bootstrap': [True, False],

#      'max_depth': [10,50,100,200,500],

#      'criterion':["gini","entropy"],

#      'max_features': ['auto', 'sqrt'],

#      'min_samples_leaf': [5,10,20,50],

#      'n_estimators': [10,30,50,100]

# }





# classifier = RandomForestClassifier()



# grid_search = GridSearchCV(classifier, param_grid, cv=3,

#                           scoring='f1', return_train_score = True)



# grid_search.fit(data_final,data_labels)
# grid_search.best_params_
# grid_search.best_score_
# cvres = grid_search.cv_results_



# for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

#     print(mean_score,params)
# rand_forest_clf = RandomForestClassifier(bootstrap=False,criterion="entropy",max_depth=100,max_features='auto',min_samples_leaf=5,n_estimators=10)

# rand_forest_clf.fit(data_final,data_labels)
# Using XGBOOST

import xgboost as xgb



xgb_cls = xgb.XGBClassifier()



# Fine Tuning

parameters = {

    "max_depth":[3,4,5,6,8,10],

    "learning_rate":[0.1,0.5,0.01],

    "n_estimators":[50,100,200,500],

    "gamma":[0,1,5]

}



grid = GridSearchCV(xgb_cls,parameters,n_jobs=-1,scoring="f1",cv=3)



grid.fit(data_final,data_labels)
grid.best_params_
grid.best_score_
test_data_final = full_pipeline.transform(test_data)
xgb_cls = xgb.XGBClassifier(gamma=1,learning_rate=0.1,max_depth=10,n_estimators=50)

xgb_cls.fit(data_final,data_labels)
# Using XGBoost

predictions = xgb_cls.predict(test_data_final)

predictions
# Make Submission

my_submission = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

my_submission.to_csv('Submission10.csv',index=False)



my_submission