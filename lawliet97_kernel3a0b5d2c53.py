import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



%matplotlib inline
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.shape
data.head(5)
#ngecek ada nilai null ga

data.isnull().values.any()
corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
data.corr()
data.head(5)
diabetes_true = len(data.loc[data['Outcome'] == True ])

diabetes_false = len(data.loc[data['Outcome'] == False ])
(diabetes_true,diabetes_false)
from sklearn.model_selection import train_test_split

feature_columns = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']

predicted_class = ["Outcome"]
X = data[feature_columns].values

y = data[predicted_class].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)
#check zero value

print("total number of rows : {0}".format(len(data)))

print("Number of rows missing Glucose:{0}".format(len(data.loc[data['Glucose'] == 0])))

print("Number of rows missing BloodPressure:{0}".format(len(data.loc[data['BloodPressure'] == 0])))

print("Number of rows missing Insulin:{0}".format(len(data.loc[data['Insulin'] == 0])))

print("Number of rows missing BMI:{0}".format(len(data.loc[data['BMI'] == 0])))

print("Number of rows missing DiabetesPedigreeFunction:{0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))

print("Number of rows missing Age:{0}".format(len(data.loc[data['Age'] == 0])))
from sklearn.preprocessing import Imputer



fill_values = Imputer(missing_values=0, strategy="mean", axis=0)



X_train = fill_values.fit_transform(X_train)

X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state=0)



random_forest_model.fit(X_train, y_train.ravel())
predict_train_data = random_forest_model.predict(X_test)



from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))
#hyper parameyer optimization

params={

    "learning_rate"    : [0.05 , 0.10 , 0.15 , 0.20 , 0.25 , 0.30 ],

    "max_depth"        : [ 3, 4, 5, 6, 8, 10 , 12 , 15],

    "min_child_weight" : [ 1, 3, 5 ,7 ],

    "gamma"            : [ 0.0, 0.1, 0.2, 0.3, 0.4],

    "colsample_bytree" : [ 0.3, 0.4, 0.5, 0.7 ]

}
#hyperparameter optimization using randomizedseachCV

from sklearn.model_selection import RandomizedSearchCV

import xgboost
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour,temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.'% (thour, tmin, round(tsec, 2)))
from datetime import datetime



start_time = timer(None)

random_search.fit(X,y.ravel())

timer(start_time)
random_search.best_estimator_
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.7, gamma=0.3,

              learning_rate=0.25, max_delta_step=0, max_depth=10,

              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X_train,y_train.ravel(),cv=10)
score
score.mean()