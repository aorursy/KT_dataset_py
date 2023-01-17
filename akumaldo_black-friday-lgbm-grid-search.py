# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline 

#allow you to plot directly without having to call .show()



import warnings

warnings.filterwarnings(action="ignore") #ignore warnings
data = pd.read_csv("../input/BlackFriday.csv")#loading the database
data.head() #quick exploration
print(data.shape) #printing the shape of our data
data.isnull().sum() #checking for null values in our data
#let's take a look at the number of unique values for each category like gender, marital_status...expected 2

data.apply(lambda x: len(x.unique()))


sns.set(style="ticks")

g = sns.FacetGrid(data, col="City_Category", col_order=['A','B','C'], aspect=0.7, height=6)

g.fig.subplots_adjust(wspace=3, hspace=.02);

g.map(sns.countplot, "Product_Category_1") #this Product category sold the least in all cities, ABC

#g.map(sns.countplot, "Product_Category_2") 

#g.map(sns.countplot, "Product_Category_3") #this Product Category sold the highest in all cities, ABC

g = sns.FacetGrid(data, col="City_Category", col_order=['A','B','C'], aspect=0.5, height=6)

g.map(plt.hist, "Purchase") #City B has the highest purchase
data.describe()
data.info()
data["Age"].value_counts() #checking this categorical column
g = sns.FacetGrid(data, col="Age", aspect=0.5, height=6)

g.map(plt.hist, "Purchase") #not surprisingly, the 26-35 has bought the most whereas 0-17 the least
g = sns.FacetGrid(data, col="City_Category", hue="Occupation", palette="GnBu_d")

g.map(plt.hist, "Purchase")

g.add_legend();
data.groupby("Product_Category_1")["Purchase"].describe()
data.groupby("Product_Category_2")["Purchase"].describe()
data.groupby("Product_Category_3")["Purchase"].describe()

corr = data.corr()

corr['Purchase'].sort_values(ascending=False)#it's not telling us much.
ax = sns.set(rc={'figure.figsize':(10,4)})

sns.heatmap(corr, annot=True).set_title('Pearsons Correlation Factors Heat Map', color='black', size='20')
sns.distplot(data['Purchase']).set_title("Purchase distribution") #purchases are concentrated within 5000 and 10000
from sklearn.model_selection import train_test_split #importing the necessary module



train_set, test_set = train_test_split(data, test_size=0.12, random_state=42) # as our dataset is rather small, gonna separate 12% to our testing set
print(train_set.shape, test_set.shape) #checking the datasets shape
train_labels = train_set["Purchase"].copy()

train = train_set.drop("Purchase", axis=1).copy()

test_labels = test_set["Purchase"].copy()

test = test_set.drop("Purchase", axis=1).copy()
print(train.shape, test.shape) #checking the shape again
#let's first implement this pipeline, which is gonna be use for numerical atributes and use the standard scaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



train_num = train.select_dtypes(exclude=['object'])

num_attribs = list(train_num)

cat_attribs = ["Age", "City_Category","Gender"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])
#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results

import time #implementing in this function the time spent on training the model

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



def modelfit(alg, dtrain, target):

    #Fit the algorithm on the data

    time_start = time.perf_counter() #start counting the time

    alg.fit(dtrain, target)

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain)

    

    cv_score = cross_val_score(alg, dtrain,target, cv=5, scoring='neg_mean_squared_error')

    cv_score = np.sqrt(-cv_score)

    

    time_end = time.perf_counter()

    

    total_time = time_end-time_start

    #Print model report:

    print("\nModel Report")

    print("RMSE : %4g" % np.sqrt(mean_squared_error(target, dtrain_predictions)))

    print("CV Score : Mean - %4g | Std - %4g | Min - %4g | Max - %4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))
train_prepared = full_pipeline.fit_transform(train)
#ok, nice, now that we have our data prepared, let's start testing some models

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

modelfit(lin_reg, train_prepared, train_labels)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

modelfit(tree_reg, train_prepared, train_labels)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

modelfit(forest_reg, train_prepared, train_labels)
from sklearn.ensemble import GradientBoostingRegressor



params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth' : 4,

            'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10, 

            'loss': 'huber', 'random_state': 5}

gdb_model = GradientBoostingRegressor(**params)

modelfit(gdb_model, train_prepared, train_labels)
test_prepared = full_pipeline.fit_transform(test)

predictions = forest_reg.predict(test_prepared) #predicting using the random forest model that has been trained.

print("RMSE : %4g" % np.sqrt(mean_squared_error(test_labels, predictions )))



cv_score = cross_val_score(forest_reg, test_prepared,test_labels, cv=20, scoring='neg_mean_squared_error')

cv_score = np.sqrt(-cv_score)



print("Cross validation score: Mean: %g" % np.mean(cv_score))
import lightgbm as lgb



lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=2000,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



modelfit(lgb_model, train_prepared, train_labels)
predictions = lgb_model.predict(test_prepared) 

print("RMSE : %4g" % np.sqrt(mean_squared_error(test_labels, predictions )))



cv_score = cross_val_score(lgb_model, test_prepared,test_labels, cv=20, scoring='neg_mean_squared_error')

cv_score = np.sqrt(-cv_score)



print("Cross validation score: Mean: %g" % np.mean(cv_score))
import xgboost as xgb

xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0.030,                 

                 learning_rate=0.07,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=1000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.95)

modelfit(xgb_model, train_prepared, train_labels)