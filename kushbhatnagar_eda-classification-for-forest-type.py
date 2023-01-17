# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the required libraries and data set 

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
dataset_train=pd.read_csv("../input/learn-together/train.csv")
#Checking first few rows

dataset_train.head()
#Total number of records

dataset_train.shape
#Column Details

dataset_train.columns
#Create feature matrix , will keep all columns except 'Id'

X=dataset_train.drop(columns=['Id'])

X.head()
sns.scatterplot(X['Elevation'],X['Aspect'],hue=X['Cover_Type'],palette='rainbow')
#Boxplot between elevation and Cover type

sns.boxplot(y=X['Elevation'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Aspect and Cover type

sns.boxplot(y=X['Aspect'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Slope and Cover type

sns.boxplot(y=X['Slope'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Horizontal_Distance_To_Hydrology and Cover type

sns.boxplot(y=X['Horizontal_Distance_To_Hydrology'],x=X['Cover_Type'],palette='rainbow')

#Boxplot between Vertical_Distance_To_Hydrology and Cover type

sns.boxplot(y=X['Vertical_Distance_To_Hydrology'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Horizontal_Distance_To_Roadways and Cover type

sns.boxplot(y=X['Horizontal_Distance_To_Roadways'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Hillshade_9am and Cover type

sns.boxplot(y=X['Hillshade_9am'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Hillshade_Noon and Cover type

sns.boxplot(y=X['Hillshade_Noon'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Hillshade_3pm and Cover type

sns.boxplot(y=X['Hillshade_3pm'],x=X['Cover_Type'],palette='rainbow')
#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type

sns.boxplot(y=X['Horizontal_Distance_To_Fire_Points'],x=X['Cover_Type'],palette='rainbow')
#Creating data frame for Degree Variables 

X_deg=X[['Elevation','Aspect','Slope','Cover_Type']]
#Creating pairplot for Degree Variables

sns.pairplot(X_deg,hue='Cover_Type')
#Creating data frame for Distance Variables 

X_dist=X[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type']]
#Creating pairplot for Degree Variables

sns.pairplot(X_dist,hue='Cover_Type')
#Creating data frame for Hillshade Variables 

X_hs=X[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]
#Creating pairplot for Hillshade Variables

sns.pairplot(X_hs,hue='Cover_Type')
#Creating data frame for Hillshade Variables 

X_wild=X[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Cover_Type']]
#Creating pairplot for Hillshade Variables

sns.pairplot(X_wild,hue='Cover_Type')
#Checking missing values 

total_missing_values_X=X.isnull().sum().sort_values(ascending=False)

total_missing_values_X
#Taking independent variable out of X and assigning to y

y=X[['Cover_Type']]

X=X.drop(columns=['Cover_Type'])
#**Commenting data scaling as scores are improved without data scaling ***

# Feature Scaling training set for better predictions 

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#X = sc.fit_transform(X)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(X,y)
# Predicting the Train set results

y_pred_lr=classifier_lr.predict(X)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_lr=confusion_matrix(y,y_pred_lr)

cm_lr
#Converting y from series to array , to generate a graph for comparision with y_pred_

y=y.values
#Converting 2 dimensional y and y_pred array into single dimension 

y=y.ravel()

y_pred_lr=y_pred_lr.ravel()
#Creating data frame for y and y_pred_ to create line plot

df_lr=pd.DataFrame({"y":y,"y_pred_lr":y_pred_lr})
#Creating scatter plot for both values to see comparision between y and y_pred

plt.figure(figsize=(25,10))

ax=sns.scatterplot(x=range(1,15121),y=df_lr['y'],color='red')

ax=sns.scatterplot(x=range(1,15121),y=df_lr['y_pred_lr'],color='blue')

ax.set_xscale('log')
# Fitting KNN classifier to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier_knn=KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)

classifier_knn.fit(X, y)
# Predicting the Train set results

y_pred_knn=classifier_lr.predict(X)
#Converting 2 dimensional  y_pred array into single dimension 

y_pred_knn=y_pred_knn.ravel()
#Creating data frame for y and y_pred_ to create line plot

df_knn=pd.DataFrame({"y":y,"y_pred_knn":y_pred_knn})
#Creating scatter plot for both values to see comparision between y and y_pred

plt.figure(figsize=(25,10))

ax=sns.scatterplot(x=range(1,15121),y=df_knn['y'],color='red')

ax=sns.scatterplot(x=range(1,15121),y=df_knn['y_pred_knn'],color='blue')

ax.set_xscale('log')
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier_rf.fit(X, y)
# Predicting the Train set results

y_pred_rf=classifier_rf.predict(X)
#Converting 2 dimensional  y_pred array into single dimension 

y_pred_rf=y_pred_rf.ravel()
#Creating data frame for y and y_pred_ to create line plot

df_rf=pd.DataFrame({"y":y,"y_pred_rf":y_pred_rf})

#Creating scatter plot for both values to see comparision between y and y_pred

plt.figure(figsize=(25,10))

ax=sns.scatterplot(x=range(1,15121),y=df_rf['y'],color='red')

ax=sns.scatterplot(x=range(1,15121),y=df_rf['y_pred_rf'],color='blue')

ax.set_xscale('log')
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies_rf = cross_val_score(estimator = classifier_rf, X = X, y = y, cv = 10)

accuracies_rf
#Calculating mean and standard deviation for random forest model

accuracies_rf.mean()

accuracies_rf.std()
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier_rf_new = RandomForestClassifier(n_estimators = 719,

                                           bootstrap=False,

                                           max_depth=464,

                                           max_features=0.3,

                                           min_samples_leaf=1,

                                           min_samples_split=2,

                                           random_state=42)

classifier_rf_new.fit(X, y)
# Predicting the Train set results

y_pred_rf_new=classifier_rf_new.predict(X)
#Converting 2 dimensional  y_pred array into single dimension 

y_pred_rf_new=y_pred_rf_new.ravel()
#Creating data frame for y and y_pred_ to create line plot

df_rf_new=pd.DataFrame({"y":y,"y_pred_rf_new":y_pred_rf_new})
#Creating scatter plot for both values to see comparision between y and y_pred

plt.figure(figsize=(25,10))

ax=sns.scatterplot(x=range(1,15121),y=df_rf_new['y'],color='red')

ax=sns.scatterplot(x=range(1,15121),y=df_rf_new['y_pred_rf_new'],color='blue')

ax.set_xscale('log')
# Applying k-Fold Cross Validation

accuracies_rf_new = cross_val_score(estimator = classifier_rf_new, X = X, y = y, cv = 10)

accuracies_rf_new
#Calculating mean and standard deviation for random forest model

accuracies_rf_new.mean()

accuracies_rf_new.std()
#importing required library and creating XGboost classifier model

#Refered above mentioned kernels for fine tuning XGB classifier model

from xgboost import XGBClassifier

classifier_xgb=XGBClassifier(n_estimators = 719,

                             max_depth = 10)

classifier_xgb.fit(X,y)
# Predicting the Train set results

y_pred_xgb=classifier_xgb.predict(X)
#Converting 2 dimensional  y_pred array into single dimension 

y_pred_xgb=y_pred_xgb.ravel()
#Creating data frame for y and y_pred_ to create line plot

df_xgb=pd.DataFrame({"y":y,"y_pred_xgb":y_pred_xgb})
#Creating scatter plot for both values to see comparision between y and y_pred

plt.figure(figsize=(25,10))

ax=sns.scatterplot(x=range(1,15121),y=df_xgb['y'],color='red')

ax=sns.scatterplot(x=range(1,15121),y=df_xgb['y_pred_xgb'],color='blue')

ax.set_xscale('log')
accuracies_xgb = cross_val_score(estimator = classifier_xgb, X = X, y = y, cv = 10)

accuracies_xgb
#Calculating mean and standard deviation for random forest model

accuracies_xgb.std()

accuracies_xgb.mean()
#Get test data 

dataset_test = pd.read_csv("../input/learn-together/test.csv")
#Create X_test and fetching id in different frame

X_test=dataset_test.drop(columns=['Id'])

y_test_id=dataset_test[['Id']]
#Converting Id into array

y_test_id=y_test_id.values
#Converting 2 dimensional y_test_id into single dimension 

y_test_id=y_test_id.ravel()
#Checking missing value in test data set

total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)

total_missing_values_X_test
#**Commenting data scaling as scores are improved without data scaling ***

#Scaling and Transforming test set also as train set is already scaled and transformed

#X_test = sc.fit_transform(X_test)
#Creating predictions from random forest model without fine tuned parameters

y_test_pred_rf=classifier_rf.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_pred_rf=y_test_pred_rf.ravel()
#Creating Submission dataframe from id and predecited Sale price

submission_df_rf=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_rf})

#Setting index as Id Column

submission_df_rf.set_index("Id")
#Converting into CSV file for submission

submission_df_rf.to_csv("submission_rf.csv",index=False)
#Creating predictions from random forest model with fine tuned parameters

y_test_pred_rf_new=classifier_rf_new.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_pred_rf_new=y_test_pred_rf_new.ravel()
#Creating Submission dataframe from id and predecited Sale price

submission_df_rf_new=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_rf_new})

#Setting index as Id Column

submission_df_rf_new.set_index("Id")
#Converting into CSV file for submission

submission_df_rf_new.to_csv("submission_rf_new.csv",index=False)
#Creating predictions from XGB model

y_test_pred_xgb=classifier_xgb.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_pred_xgb=y_test_pred_xgb.ravel()
#Creating Submission dataframe from id and predecited Sale price

submission_df_xgb=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_xgb})

#Setting index as Id Column

submission_df_xgb.set_index("Id")
#Converting into CSV file for submission

submission_df_xgb.to_csv("submission_xgb.csv",index=False)