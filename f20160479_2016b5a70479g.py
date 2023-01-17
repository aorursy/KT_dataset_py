import pandas as pd

import numpy as np

from sklearn import svm

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from sklearn import linear_model as lm

from sklearn.model_selection import train_test_split

%matplotlib inline
train_file = r"../input/eval-lab-1-f464-v2/train.csv"

test_file = r"../input/eval-lab-1-f464-v2/test.csv"
df_train = pd.read_csv(train_file)

df_test = pd.read_csv(test_file)
#function to check for nulls and unique values in each column

def detail_df(df):

    data_type = pd.concat([df.dtypes,df.nunique(),df.isnull().sum()],axis=1)

    data_type.columns = ["dtype", "unique","no of null"]

    return data_type

df_detail = detail_df(df_train)

df_detail
#filling up missing values

df_train.fillna(value=df_train.mean(),inplace=True)

df_train.fillna(value=df_train.mode().loc[0],inplace=True)



df_test.fillna(value=df_train.mean(),inplace=True)

df_test.fillna(value=df_train.mode().loc[0],inplace=True)
#Plotting Correlation between features

plt.figure(figsize =(10,10))

sns.heatmap(data = df_train.corr(),annot = True)
#columns to be dropped: "proc1"-after dropping

# = "feature11","feature10","feature9","feature8","feature4","feature2","feature1"

drop_columns = []



#dropping columns

df_proc1_train = df_train.drop(labels = drop_columns,axis =1,inplace = False)

df_detail_train = detail_df(df_proc1_train)

#df_detail_train



df_proc1_test = df_test.drop(labels = drop_columns,axis =1,inplace = False)



#########

df_proc1_test["type"].replace("new", 1, inplace = True)

df_proc1_test["type"].replace("old", -1, inplace = True)

df_proc1_train["type"].replace("new", 1, inplace = True)

df_proc1_train["type"].replace("old", -1, inplace = True)

#########



df_detail_test = detail_df(df_proc1_test)

df_detail_test

numerical_features = ["feature3","feature5","feature6","feature7","feature11","feature10","feature9","feature8","feature4","feature2","feature1"]

categorical_features = ["type"]



X_train = df_proc1_train[numerical_features+categorical_features]

y_train = df_proc1_train["rating"]



X_test = df_proc1_test[numerical_features+categorical_features]

#Y_train = df_proc1_test["price"]



#scaling

scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_test[numerical_features] = scaler.transform(X_test[numerical_features])

Xt_train,X_val,yt_train,y_val = train_test_split(X_train,y_train,test_size=0.02,random_state=42)  #Checkout what does random_state do
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (accuracy_score,mean_squared_error)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



#TODO

clf1 = ExtraTreesClassifier(bootstrap=True,class_weight='balanced')        #Initialize the classifier object



parameters = {



    'max_depth': [27],

    'n_estimators':[485],



             }    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf1,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(Xt_train,yt_train)        #Fit the gridsearch object with X_train,y_train



best_clf1 = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf1.fit(Xt_train, yt_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf1.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
np.sqrt(mean_squared_error(best_clf1.predict(X_val),y_val))
Xt_train1,X_proc,yt_train1,y_proc = train_test_split(X_train,y_train,test_size=0.4,random_state=42)

Xt_train2,X_val,yt_train2,y_val = train_test_split(X_proc,y_proc,test_size=0.1,random_state=42)
from sklearn.ensemble import ExtraTreesRegressor

clf1 = ExtraTreesRegressor(warm_start=True,n_estimators=100)        #Initialize the classifier object



clf1.fit(Xt_train1,yt_train1)



clf1.fit(Xt_train2,yt_train2)









np.sqrt(mean_squared_error(clf1.predict(X_val),y_val))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



#TODO

clf1 = RandomForestClassifier(bootstrap=False,class_weight='balanced')        #Initialize the classifier object



parameters = {

    'max_depth': [27],

    'max_features':['auto'],



    'n_estimators':[485],



             }    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf1,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(Xt_train,yt_train)        #Fit the gridsearch object with X_train,y_train



best_clf1 = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf1.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf1.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))

print(np.sqrt(mean_squared_error(best_clf1.predict(X_val),y_val)))