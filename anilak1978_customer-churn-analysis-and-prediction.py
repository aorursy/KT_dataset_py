# Import initial libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/anilak1978/customer-churn/master/bigml_59c28831336c6604c800002a.csv")
df.head()
# checking for missing values

df.isnull().sum().values.sum()
# for loop to see unique values

for column in df.columns.values.tolist():

    print(column)

    print(df[column].unique())

    print("")
# check data types

df.dtypes
# update churn data type and boolen values to 0 and 1

df["churn"]=df["churn"].astype("str")
df["churn"]=df["churn"].replace({"False":0, "True":1})
# look at the brief overview of the data

df.info()
# look at statistical information

df.describe()
# group the data to see churn rate by state

df_state = df.groupby("state")["churn"].mean().reset_index()
plt.figure(figsize=(20,5))

sns.barplot(x="state", y="churn", data=df_state)
# look at churn rate for all categorical variables

categorical_variables = ["area code", "international plan", "voice mail plan", "state"]

for i in categorical_variables:

    data=df.groupby(i)["churn"].mean().reset_index()

    plt.figure(figsize=(20,5))

    sns.barplot(x=data[i], y="churn", data=data)
# look at the distribution of categorical variables

for i in categorical_variables:

    plt.figure(figsize=(20,5))

    sns.countplot(x=df[i], data=df)
plt.figure(figsize=(10,5))

sns.countplot(x=df["churn"], data=df)
# Analysing numerical variables

numerical_variables=["account length", "number vmail messages", "total day minutes", "total day charge", "total day calls", 

                     "total day charge", "total eve minutes", "total eve charge", "total night minutes",

                    "total intl minutes", "total intl calls",

                    "total intl charge", "customer service calls"]
# looking at relationship for each numerical variable and churn

for i in numerical_variables:

    plt.figure(figsize=(20,5))

    sns.regplot(x=df[i], y="churn", data=df)
# looking at correlation within numerical variables

corr=df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr, annot=True)
# feature selection

X = df[["account length", "international plan", "total day charge", "total night charge", "total intl charge", "customer service calls", "state"]]
# target selection

y =df["churn"]
# review feature set

X[0:5]
# update state with one hot coding

X=pd.get_dummies(X, columns=["state"])
# make sure i am using feature set values 

X=X.values
# preprocess to update str variables to numerical variables

from sklearn import preprocessing

international_plan=preprocessing.LabelEncoder()

international_plan.fit(["no", "yes"])

X[:,1] = international_plan.transform(X[:,1])
# create training and testing set

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)
#create model using random forest classifier and fit the training set

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_trainset, y_trainset)
#create prediction using the model

rf_pred = rf_model.predict(X_testset)

rf_pred[0:5]
# Looking at the accuracy score (using two methods)

from sklearn import metrics

rf_model.score(X_testset, y_testset)

metrics.accuracy_score(y_testset, rf_pred)
# confusion matrics to find precision and recall

from sklearn.metrics import confusion_matrix

confusion_matrix(y_testset, rf_pred)
# Looking at the precision score

from sklearn.metrics import precision_score

precision_score(y_testset, rf_pred)
# Looking at the recall score

from sklearn.metrics import recall_score

recall_score(y_testset, rf_pred)
# find probability for each prediction

prob=rf_model.predict_proba(X_testset)[:,1]
# look at ROC curve, which gives us the false and true positive predictions

from sklearn.metrics import roc_curve

fpr, tpr, thresholds=roc_curve(y_testset, prob)

plt.plot(fpr, tpr)
# Looking at the area under the curve

from sklearn.metrics import roc_auc_score

auc=roc_auc_score(y_testset, prob)

auc
#looking at the f1_score

from sklearn.metrics import f1_score

f1_score(y_testset, rf_pred)
#Looking at the best possible estimator

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators': np.arange(10,51)}

rf_cv=GridSearchCV(RandomForestClassifier(), param_grid)

rf_cv.fit(X,y)

rf_cv.best_params_
# looking at the best feature score

rf_cv.best_score_
# looking at the importance of each feature

importances=rf_model.feature_importances_
# visualize to see the feature importance

indices=np.argsort(importances)[::-1]

plt.figure(figsize=(20,10))

plt.bar(range(X.shape[1]), importances[indices])

plt.show()
# creating the svm model and fitting training set

# make sure to update probability to True for proabbility evaluation

from sklearn.svm import SVC

svc_model=SVC(probability=True)

svc_model.fit(X_trainset, y_trainset)
# creating the svm prediction

svc_pred=svc_model.predict(X_testset)

svc_pred[0:5]
# look at the accuracy score

svc_model.score(X_testset, y_testset)
# Look at the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_testset, svc_pred)
#precision score for svm

precision_score(y_testset, svc_pred)
# recall score for svm

recall_score(y_testset, svc_pred)
# probability for each prediction

prob_2=svc_model.predict_proba(X_testset)[:,1]
# look at ROC curve

fpr, tpr, thresholds=roc_curve(y_testset, prob_2)

plt.plot(fpr, tpr)
# area under the curve

auc=roc_auc_score(y_testset, prob)

auc
# find ideal degree for SVM model

param_grid_2={'degree': np.arange(1,50)}

svc_cv=GridSearchCV(SVC(), param_grid_2)

svc_cv.fit(X,y)

svc_cv.best_params_