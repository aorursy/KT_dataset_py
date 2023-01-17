import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
print("The shape of the train set is {}".format(train.shape))
print("The shape of the test set is {}".format(test.shape))
train.dtypes
test.dtypes
train.head(5)
test.head(5)
train.isnull().sum()
test.isnull().sum()
train = train.drop(["id"],axis=1)
test_id = test["id"]
test= test.drop(["id"],axis=1)
numerical_columns = train.select_dtypes(include = ("int64","float64"))
numerical_columns.describe().T
plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
graph = sns.countplot(x = "Response",data=train).set_title("Response Count")

plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
graph = sns.countplot(x = "Response",data=train,hue="Gender").set_title("Response vs Gender")

plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
graph = sns.countplot(x = "Response",data=train,hue="Vehicle_Age").set_title("Response vs Vehicle Age")

plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
graph = sns.distplot(train.Age,color="red").set_title("Age Distribution of Customers")
plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
g = sns.FacetGrid(train, col='Response', height=5)
g.map(plt.hist, "Age",color="red");

plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
graph = sns.distplot(train.Annual_Premium,color="red").set_title("Annual Premium Distribution of Customers")
plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
g = sns.FacetGrid(train, col='Response', height=5)
g.map(plt.hist, "Annual_Premium",color="red");

plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
g = sns.FacetGrid(train, col='Vehicle_Age', height=5)
g.map(plt.hist, "Annual_Premium",color="red");
y = train.Response.values
train.drop(["Response"],axis=1,inplace = True)
fulldata = pd.concat([train,test],axis=0)
print("the shape of fulldata is {}".format(fulldata.shape))
fulldata.dtypes
numerical_columns = fulldata.select_dtypes(include=("int64","float64"))
categorical_columns = fulldata.select_dtypes(include=("object"))
from sklearn.preprocessing import LabelEncoder
for column in categorical_columns:
    fulldata[column] = LabelEncoder().fit_transform(fulldata[column])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for column in numerical_columns:
    fulldata[column] = scaler.fit_transform(fulldata[[column]])
fulldata.head(5)
train_len = len(train)
train = fulldata[:train_len]
test = fulldata[train_len:]

train.shape
test.shape
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size = 0.25,random_state=52)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier().fit(x_train,y_train)
lightgbm = LGBMClassifier().fit(x_train,y_train)
randomforest = RandomForestClassifier().fit(x_train,y_train)
neural = MLPClassifier().fit(x_train,y_train)
models = [xgb,lightgbm,randomforest,neural]

for model in models:
    model_name = model.__class__.__name__
    pred = model.predict(x_test)
    acc = accuracy_score(pred,y_test)
    print(model_name + " ---> " + " accuracy : {:.2%} ".format(acc))

