# Importing time

import time



# Importing Markdown(styling) and display 

from IPython.display import display,Markdown



# Importing libraries. 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# sklearn libraries

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import cross_val_predict



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



# warnings ignore

import warnings

warnings.filterwarnings("ignore")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")



# view First 5 rows with all cols of train data.

train_df.head()
# view First 5 rows with all cols of test data.  

test_df.head()
# Check our data is balanced or imbalanced.

sns.countplot(x="Survived",data=train_df)

# Count of each class

train_df.Survived.value_counts()
# Create Target variable and pass the Survived col

y=train_df["Survived"].values

# Drop Target variable in train data

train_df.drop(["Survived","PassengerId"],inplace=True,axis=1)

test_df.drop(["PassengerId"],inplace=True,axis=1)

train_df.head()
# create new variable train and pass "1" for train data and "0" for test data to retrive test data in future.

train_df["train"]=1

test_df["train"]=0



# It's better to combine both train and test data using pandas concat function.

combined_df=pd.concat([train_df,test_df])
# view combined data.

combined_df.head()
# Check shape of combined data.

combined_df.shape
from IPython.display import Image

Image("/kaggle/input/missing-values-mechanism/Missingtheory.png")
# Function to get null values info.

def null_info(data,string):

        data_cols=data.columns[data.isna().any()]

        null_sum=data[data_cols].isna().sum().sort_values(ascending=False)

        null_dtype=data[data_cols].dtypes

        null_percent=round(data[data_cols].isna().mean()*100,2)

        concat=pd.concat([null_sum,null_dtype,null_percent],axis=1,keys=["NaN count","Dtypes","%Ages"],sort=True)

        display(Markdown("### `{}`".format(string)))

        display(concat.head()) 

        

        time.sleep(.5)



        if any(concat["%Ages"] > 50):

            count=sum(concat["%Ages"] > 50)

            index_Count=concat[concat["%Ages"]>50].index.tolist()

            display(Markdown("Found `{}` columns **{}** which have more than 50% of missing values in `{}` ".format(count,index_Count,string)))

            print("Removing columns which have more than 50% missing values")

            print(" ")

            concat.drop(index_Count,inplace=True,axis=0)

            display(Markdown(">**Removed** columns `{}` from data".format(index_Count)))

        display(concat)

        return (concat.index,index_Count)
# Null_info function returns cols that are present and cols are deleted.

null_df,del_rows=null_info(combined_df,"Null values on train Data")
# view cols which are not removed by null_info function.

null_df
# create new dataframe and add those cols with values, because we don't want to spoil the original dataframes

# by adding,removing, modifying values and doing mutiple operations.

model_df= combined_df[null_df]
# view new dataframe

model_df.head()
# Get count of missing values

model_df.isna().sum()
from IPython.display import Image

Image("/kaggle/input/missing-values-mechanism/Missingtheory.png")
# function for imputation and display result.head of 2.

def imputation(data):

    data_numeric=data.select_dtypes(include=np.number)

    data_categorical=data.select_dtypes(exclude=np.number)

    display(Markdown("## Before imputation"))

    display(data[data_numeric.isna().values].head(2))

    display(data[data_categorical.isna().values].head(2))

    data.fillna(data_numeric.median(),inplace=True) # imputing with median robust to outliers.

    for i in data_categorical.columns:

        data[i].fillna(data[i].value_counts().index[0], inplace=True) # imputing categorical with mode (most frequent).

    display(Markdown("## After imputation"))

    display(data[data_numeric.isna().values].head(2))

    display(data[data_categorical.isna().values].head(2))
imputation(model_df)
def remove_exists(data1,data2,del_rows):

    data1.drop(data2.columns,axis=1,inplace=True)

    data1.drop(del_rows,axis=1,inplace=True)

    display(data1.head())

remove_exists(combined_df,model_df,del_rows)
# Get Numeric and Categorical Values into separate

def numr_cate(data):

    data_numr=data.select_dtypes(include=np.number)

    data_cate=data.select_dtypes(exclude=np.number)

    display(Markdown("## Numeric Columns"))

    display(data_numr.head())

    display(Markdown("## Categorical Columns"))

    display(data_cate.head())

    return data_numr,data_cate

df_num,df_cate=numr_cate(combined_df)
df_num.head()
df_cate.head()
# Drop ticket column

df_cate.drop("Ticket",axis=1,inplace=True)
# combine categorical and numeric dataframe to model_df.

model_df[df_cate.columns]=df_cate.copy()

model_df[df_num.columns]=df_num.copy()
model_df.head()
model_df.head()
model_df["Sex"]=model_df.Sex.map({"male":0,"female":1})
model_df["Embarked"]=model_df.Embarked.map({"S":0,"C":1,"Q":2})
# combine SibSp and Parch to one variable called family

model_df["family"]=model_df["SibSp"] + model_df["Parch"] + 1 
# Titles  

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

# extract titles

model_df['Title'] = model_df.Name.str.extract(' ([A-Za-z]+)\.', expand= False)

# replace titles with a more common title or as Rare

model_df['Title'] = model_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

model_df['Title'] = model_df['Title'].replace('Mlle', 'Miss')

model_df['Title'] = model_df['Title'].replace('Ms', 'Miss')

model_df['Title'] = model_df['Title'].replace('Mme', 'Mrs')

# convert titles into numbers

model_df['Title'] = model_df['Title'].map(titles)

# filling NaN with 0, to get safe

model_df['Title'] = model_df['Title'].fillna(0)
# Binning age with certain values

model_df['Age'] = model_df['Age'].astype(int)

model_df.loc[ model_df['Age'] <= 11, 'Age'] = 0

model_df.loc[(model_df['Age'] > 11) & (model_df['Age'] <= 18), 'Age'] = 1

model_df.loc[(model_df['Age'] > 18) & (model_df['Age'] <= 22), 'Age'] = 2

model_df.loc[(model_df['Age'] > 22) & (model_df['Age'] <= 27), 'Age'] = 3

model_df.loc[(model_df['Age'] > 27) & (model_df['Age'] <= 33), 'Age'] = 4

model_df.loc[(model_df['Age'] > 33) & (model_df['Age'] <= 40), 'Age'] = 5

model_df.loc[(model_df['Age'] > 40) & (model_df['Age'] <= 66), 'Age'] = 6

model_df.loc[ model_df['Age'] > 66, 'Age'] = 6
model_df['Fare'] = model_df['Fare'].astype(int)

# Binning Fare 

model_df.loc[ model_df['Fare'] <= 7.91, 'Fare'] = 0

model_df.loc[(model_df['Fare'] > 7.91) & (model_df['Fare'] <= 14.454), 'Fare'] = 1

model_df.loc[(model_df['Fare'] > 14.454) & (model_df['Fare'] <= 31), 'Fare'] = 2

model_df.loc[(model_df['Fare'] > 31) & (model_df['Fare'] <= 99), 'Fare'] = 3

model_df.loc[(model_df['Fare'] > 99) & (model_df['Fare'] <= 250), 'Fare'] = 4

model_df.loc[ model_df['Fare'] > 250, 'Fare'] = 5
# dropping extra columns 

model_df.drop(["SibSp","Parch","Name"],axis=1,inplace=True)

model_df.head()
# Divide train and test data 

X_train=model_df[model_df.train==1]

X_test=model_df[model_df.train==0]

Y_train=y # target variable

X_train.drop("train",axis=1,inplace=True)

X_test.drop("train",axis=1,inplace=True)
X_train.head()
X_test.head()
X_train.shape,X_test.shape
# SGD Classifier

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# Random tree classifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# Gradient boosting

GBC = GradientBoostingClassifier()

GBC.fit(X_train, Y_train)

acc_GBC = round(GBC.score(X_train, Y_train) * 100, 2)
# Logistic regression.

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# Guassian navie bayes.

gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train) 

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# Linear support vector classifier.

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# Decision Tree

decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_train, Y_train)* 100, 2)
# Table for scores

results = pd.DataFrame({

'Model': ['Support Vector Machines','GBC', 'KNN', 'Logistic Regression',

'Random Forest', 'Naive Bayes', 'Perceptron',

'Stochastic Gradient Decent',

'Decision Tree'],

'Score': [acc_linear_svc,acc_GBC, acc_knn, acc_log,

acc_random_forest, acc_gaussian, acc_perceptron,

acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(10)
# select random forest

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

# Cross validation == n * (train_test_split) and scoring 

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
# Feature importances with random forest

importances =pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
# Random Forest

random_forest = RandomForestClassifier(criterion = "gini",

                                        min_samples_leaf = 1,

                                        min_samples_split = 10,

                                        n_estimators=100,

                                        max_features='auto',

                                        oob_score= True,

                                        random_state=1,

                                        n_jobs=-1)

random_forest.fit(X_train, Y_train)

print("oob score:", round(random_forest.score(X_train,Y_train), 4)*100, "%")
# Evaluation of model using classification metrics

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

# cross validation prediction 

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=10)

confusion_matrix(Y_train, predictions)
# Precision and recall score 

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
# Getting the probabilities of our predictions

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]
# Roc_Auc score 

from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(Y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
# make prediction and ready for submitting

predictionss=random_forest.predict(X_test)
# make submission file for kaggle

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

submission =pd.DataFrame({'PassengerId':test_df.PassengerId,'Survived':predictionss})

submission.head()
# check for shape

submission.shape
# save to csv file

submission.to_csv("submission.csv",index=False)