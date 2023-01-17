# import necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
# import data modelling libraries

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# load the dataset

data= pd.read_csv("../input/banking-dataset-classification/new_train.csv")



# check shape of dataset

print("shape of the data:", data.shape)

data.head()
# check data types of all columns

data.dtypes
data.isnull().sum()
# target class count

data["y"].value_counts()
sns.countplot(data["y"])

plt.title("target variable")
# percentage of class present in target variable(y) 

print("percentage of NO and YES\n",data["y"].value_counts()/len(data)*100)
# indentifying the categorical variables

cat_var= data.select_dtypes(include= ["object"]).columns

print(cat_var)



# plotting bar chart for each categorical variable

plt.style.use("ggplot")



for column in cat_var:

    plt.figure(figsize=(20,4))

    plt.subplot(121)

    data[column].value_counts().plot(kind="bar")

    plt.xlabel(column)

    plt.ylabel("number of customers")

    plt.title(column)
# replacing "unknown" with the mode

for column in cat_var:

    mode= data[column].mode()[0]

    data[column]= data[column].replace("unknown", mode)
# indentifying the numerical variables

num_var= data.select_dtypes(include=np.number)

num_var.head()
# plotting histogram for each numerical variable

plt.style.use("ggplot")

for column in ["age", "duration", "campaign"]:

    plt.figure(figsize=(20,4))

    plt.subplot(121)

    sns.distplot(data[column], kde=True)

    plt.title(column)
data.drop(columns=["pdays", "previous"], axis=1, inplace=True)
plt.style.use("ggplot")

for column in cat_var:

    plt.figure(figsize=(20,4))

    plt.subplot(121)

    sns.countplot(data[column], hue=data["y"])

    plt.title(column)    

    plt.xticks(rotation=90)
data.describe()
# compute interquantile range to calculate the boundaries

lower_boundries= []

upper_boundries= []

for i in ["age", "duration", "campaign"]:

    IQR= data[i].quantile(0.75) - data[i].quantile(0.25)

    lower_bound= data[i].quantile(0.25) - (1.5*IQR)

    upper_bound= data[i].quantile(0.75) + (1.5*IQR)

    

    print(i, ":", lower_bound, ",",  upper_bound)

    

    lower_boundries.append(lower_bound)

    upper_boundries.append(upper_bound)
lower_boundries
upper_boundries
# replace the all the outliers which is greater then upper boundary by upper boundary

j = 0

for i in ["age", "duration", "campaign"]:

    data.loc[data[i] > upper_boundries[j], i] = int(upper_boundries[j])

    j = j + 1  
# without outliers

data.describe()
#categorical features

cat_var
# check categorical class

for i in cat_var:

    print(i, ":", data[i].unique())
# initializing label encoder

le= LabelEncoder()



# iterating through each categorical feature and label encoding them

for feature in cat_var:

    data[feature]= le.fit_transform(data[feature])
# label encoded dataset

data.head()
# feature variables

x= data.iloc[:, :-1]



# target variable

y= data.iloc[:, -1]
plt.figure(figsize=(15,7))

sns.heatmap(data.corr(), annot=True)
#initialising oversampling

smote= SMOTETomek(0.75)



#implementing oversampling to training data

x_sm, y_sm= smote.fit_sample(x,y)



# x_sm and y_sm are the resampled data



# target class count of resampled dataset

y_sm.value_counts()
x_train, x_test, y_train, y_test= train_test_split(x_sm, y_sm, test_size=0.2, random_state=42)
# selecting the classifier

log_reg= LogisticRegression()



# selecting hyperparameter tuning

log_param= {"C": 10.0**np.arange(-2,3), "penalty": ["l1", "l2"]}



# defining stratified Kfold cross validation

cv_log= StratifiedKFold(n_splits=5)



# using gridsearch for respective parameters

gridsearch_log= GridSearchCV(log_reg, log_param, cv=cv_log, scoring= "f1_macro", n_jobs=-1, verbose=2)



# fitting the model on resampled data

gridsearch_log.fit(x_train, y_train)



# printing best score and best parameters

print("best score is:" ,gridsearch_log.best_score_)

print("best parameters are:" ,gridsearch_log.best_params_)
# checking model performance

y_predicted= gridsearch_log.predict(x_test)



cm= confusion_matrix(y_test, y_predicted)

print(cm)

sns.heatmap(cm, annot=True)

print(accuracy_score(y_test, y_predicted))

print(classification_report(y_test, y_predicted))
# random forest

rf= RandomForestClassifier()



rf_param= { 

           "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],

           "max_features": ["auto", "sqrt", "log2"],

#            "max_depth": [4,5,6,7,8],

           "max_depth": [int(x) for x in np.linspace(start=5, stop=30, num=6)],

           "min_samples_split": [5,10,15,100],

           "min_samples_leaf": [1,2,5,10],

           "criterion":['gini', 'entropy'] 

          }



cv_rf= StratifiedKFold(n_splits=5)



randomsearch_rf= RandomizedSearchCV(rf, rf_param, cv=cv_rf, scoring= "f1_macro", n_jobs=-1, verbose=2, n_iter=10)



randomsearch_rf.fit(x_train, y_train)



print("best score is:", randomsearch_rf.best_score_)

print("best parameters are:", randomsearch_rf.best_params_)
# checking model performance

y_predicted_rf= randomsearch_rf.predict(x_test)



print(confusion_matrix(y_test, y_predicted_rf))

sns.heatmap(confusion_matrix(y_test, y_predicted_rf), annot=True)

print(accuracy_score(y_test, y_predicted_rf))

print(classification_report(y_test, y_predicted_rf))
test_data= pd.read_csv("../input/banking-dataset-classification/new_test.csv")

test_data.head()
# predicting the test data

y_predicted= randomsearch_rf.predict(test_data)

y_predicted
# dataset of predicted values for target variable y

prediction= pd.DataFrame(y_predicted, columns=["y_predicted"])

prediction_dataset= pd.concat([test_data, prediction], axis=1)

prediction_dataset