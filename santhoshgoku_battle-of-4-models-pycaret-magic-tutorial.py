!pip install quickdataanalysis==0.0.7
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import metrics

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from sklearn.pipeline import make_pipeline

from quickdataanalysis import data_analysis as qd
root_path = '/kaggle/input/titanic/'
train = os.path.join(root_path,'train.csv')

test = os.path.join(root_path,"test.csv")

# submission = os.path.join(root_path,'gender_submission.csv')
df_train = pd.read_csv(train)

df_test = pd.read_csv(test)

# df_submission = pd.read_csv(subission)
df_train.head()
df_train.info()
df_train.describe()
drop_cols = ["Cabin","Embarked","Ticket","Name","PassengerId"]

df_train = df_train.drop(drop_cols,axis=1)

df_test = df_test.drop(drop_cols,axis=1)
fig, ax = plt.subplots(figsize=(20,10)) 

df_train_corr = df_train.corr()

sns.heatmap(df_train_corr,xticklabels=df_train_corr.columns,yticklabels=df_train_corr.columns,annot=True,ax=ax)
df_train["Sex"].value_counts().plot.bar(rot=0,color={"red","blue"},title="gender distrbution")
#creating dummies for train dataset

cols = ["Sex"]

df_train = qd.create_dummies(df_train,cols)
#creating dummies for test dataset

df_test = qd.create_dummies(df_test,cols)
df_train['Age'] = df_train['Age'].interpolate()

df_test['Fare'] = df_test['Fare'].interpolate()

df_test['Age'] = df_test['Age'].interpolate()
total_male = qd.column_value_count(df_train["male"] ,1 )

total_female = qd.column_value_count(df_train["female"] ,1)

print(f"Total male onboard = {total_male} \n \nTotal female onboard = {total_female}")
#getting the male & female surviver count

survived_female = qd.column_value_count(df_train[((df_train["female"]== 1) & (df_train["Survived"]==1))]["female"],1)

survived_male = qd.column_value_count(df_train[((df_train["male"]== 1) & (df_train["Survived"]==1))]["male"],1)
plot = df_train["female"].value_counts().plot.pie(figsize=(11, 6),title=f"no of female survived is {survived_female}",legend=True)
plot = df_train["male"].value_counts().plot.pie(figsize=(11, 6),title=f"no of male survived is {survived_male}",legend=True)
pclass1 = qd.column_value_count(df_train[((df_train["Pclass"]==1) & (df_train["Survived"]==1) )]["Pclass"],1)

pclass2 = qd.column_value_count(df_train[((df_train["Pclass"]==2) & (df_train["Survived"]==1) )]["Pclass"],2)

pclass3 = qd.column_value_count(df_train[((df_train["Pclass"]==3) & (df_train["Survived"]==1) )]["Pclass"],3)
fig, axes = plt.subplots(nrows=1, ncols=3)

fig.tight_layout(w_pad = 5,pad =1)

((df_train["Pclass"]==1) & (df_train["Survived"]==1) ).value_counts().plot.bar(legend=True,color="orange",title=f"Passenger Class 1 - {pclass1}",ax=axes[0])

((df_train["Pclass"]==2) & (df_train["Survived"]==1) ).value_counts().plot.bar(legend=True,color="red",title=f"Passenger Class 2 - {pclass2}",ax=axes[1])

((df_train["Pclass"]==3) & (df_train["Survived"]==1) ).value_counts().plot.bar(legend=True,color="blue",title=f"Passenger Class 3 - {pclass3}",ax=axes[2])
#seperating the target variable from the dataset

x = df_train.drop("Survived",axis=1)

y = df_train["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
#list of model that we will be using

knn = KNeighborsClassifier()

svm = SVC()

random_forest = RandomForestClassifier()

regression = LogisticRegression(max_iter=10000)
x,y = x_train,y_train

knn.fit(x,y)

svm.fit(x,y)

random_forest.fit(x,y)

regression.fit(x,y)
Y_pred_knn = knn.predict(x_test)

Y_pred_svm = svm.predict(x_test)

Y_pred_random_forest = random_forest.predict(x_test)

Y_pred_regression = regression.predict(x_test)
print("Knn Preformance")

print(metrics.classification_report(y_test, Y_pred_knn))
print("svm Preformance")

print(metrics.classification_report(y_test, Y_pred_svm))
print("Random Forest Preformance")

print(metrics.classification_report(y_test, Y_pred_random_forest))
print("Logistic Regression Preformance")

print(metrics.classification_report(y_test, Y_pred_regression))
!pip install pycaret
from pycaret.classification import *
clf1 = setup(df_train, target ='Survived', log_experiment = True, experiment_name = 'titanic survivor prediction')
# compare all baseline models and select top 5

top5 = compare_models(n_select = 5)
# tune top 5 base models

tuned_top5 = [tune_model(i) for i in top5]
# ensemble top 5 tuned models

bagged_top5 = [ensemble_model(i) for i in tuned_top5]
cat = create_model("catboost")
cat_pred_new = predict_model(cat, data = df_test)
rf = create_model("rf")
plot_model(rf)
plot_model(rf,plot="learning")
plot_model(rf,plot="feature")
plot_model(rf,plot="class_report")
df_test_submit = pd.read_csv(test)

res = pd.DataFrame({"PassengerId":df_test_submit["PassengerId"],"Survived":cat_pred_new['Label']})

res.to_csv("submission.csv",index=False)