# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import datetime

%matplotlib inline



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore")



import types

import pandas as pd



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#let's open the second version

data = pd.read_csv('../input/Admission_Predict.csv') #loading the dataframe
data.head() #taking a peek at the first 5 entries
data.set_index("Serial No.").head() #it's only logical to use the serial No. as the index for this dataset
corr = data.corr() #saving the correlation for later use

ax = sns.set(rc={'figure.figsize':(10,4)})

sns.heatmap(corr, annot=True).set_title('Pearsons Correlation Factors Heat Map', color='black', size='20')
#let's take a look at those values distribution in our dataset

sns.countplot(data['CGPA'].value_counts()) #from 1 to 9, higher the better
sns.distplot(data['GRE Score'])
#Gonna implement a function to help us look for null values, showing the relative percentage

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : nulldata})

    return ratio_missing_data.head(30)
missing_values_calculate(data)
from sklearn.model_selection import train_test_split #importing the necessary module



train_set, test_set = train_test_split(data, test_size=0.2, random_state=42) 
print(train_set.shape, test_set.shape) #printing the shape of the data
train = train_set.drop(['Chance of Admit ','Serial No.'], axis=1).copy()

train_serial = train_set['Serial No.'].copy()

train_labels = train_set['Chance of Admit '].copy()

test = test_set.drop(['Chance of Admit ','Serial No.'], axis=1).copy()

test_labels = test_set['Chance of Admit '].copy()

test_serial = test_set['Serial No.'].copy()
print(train.shape, train_labels.shape, test.shape, test_labels.shape) #printing the shape of our data after having drop and copied the relevant columns
#this pipeline is gonna be use for numerical atributes and standard scaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])
#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import KFold,StratifiedKFold,



def modelfit(alg, dtrain, target):

    #Fit the algorithm on the data

    alg.fit(dtrain, target)

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain)

    

    #if you wanna use StratifiedKfold or Kfold, use the following code

    #folds = 3 #number of folds to be used during our stratifieldKfold

    #skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    

    cv_score = cross_val_score(alg, dtrain,target, cv=5, scoring='neg_mean_squared_error')

    cv_score = np.sqrt(-cv_score)

    

    #Print model report:

    print("\nModel Report")

    print("RMSE : %4g" % np.sqrt(mean_squared_error(target, dtrain_predictions)))

    print("CV Score : Mean - %4g | Std - %4g | Min - %4g | Max - %4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    
train_prepared = num_pipeline.fit_transform(train)
#ok, nice, now that we have our data prepared, let's start testing some models

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

modelfit(lin_reg, train_prepared, train_labels)
from sklearn.kernel_ridge import KernelRidge



model_kernel = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

modelfit(model_kernel, train_prepared, train_labels)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=100, random_state=42) 

modelfit(forest_reg, train_prepared, train_labels)
from sklearn.ensemble import GradientBoostingRegressor



params = {'n_estimators': 3000, 'learning_rate': 0.05, 'max_depth' : 4,

            'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10, 

            'loss': 'huber', 'random_state': 5}

gdb_model = GradientBoostingRegressor(**params)

#modelfit(gdb_model, train_prepared, train_labels)

modelfit(gdb_model,train,train_labels)
# Plot feature importance

feature_importance = gdb_model.feature_importances_

# make importances relative to max importance

plt.figure(figsize=(40, 40)) #figure size

feature_importance = 100.0 * (feature_importance / feature_importance.max()) #making it a percentage relative to the max value

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, train.columns[sorted_idx], fontsize=15) #used train_drop here to show the name of each feature instead of our train_prepared 

plt.xlabel('Relative Importance', fontsize=20)

plt.ylabel('Features', fontsize=20)

plt.title('Variable Importance', fontsize=30)
#this function is gonna be used when it has been estimated the best features, eg: using random forest, in our case Gradient Boosting

## then we would want to especify only those features when we train our model.

from sklearn.base import BaseEstimator, TransformerMixin



def indices_of_top_k(arr, k):

    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
#k is the number of features you want.

k= 4 # for this dataset...

preparation_and_feature_selection_pipeline = Pipeline([

    ('preparation', num_pipeline),

    ('feature_selection', TopFeatureSelector(feature_importance, k))

])
#test_prepared = num_pipeline.transform(test) #transforming the testing and scaling it ir order to use our model

final_predictions = gdb_model.predict(test) #Gradient Boosting model



final_mse = mean_squared_error(test_labels, final_predictions)

final_rmse = np.sqrt(final_mse)
print(final_mse)
#importing relevant modules for this part,gonna create a visual representation

import shap #for SHAP values



def chance_admit(model, example):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(example)

    shap.initjs()

    return shap.force_plot(explainer.expected_value, shap_values, example)
#let's get a patience and take a look at the shap force plot.

p = 2 #helping variable, easily specify different patients.

data_for_prediction = test.iloc[p,:].astype(float)

chance_admit(gdb_model, data_for_prediction)