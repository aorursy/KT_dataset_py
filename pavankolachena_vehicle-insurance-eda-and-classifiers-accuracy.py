#importing the Libraries
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

#from plot_metric.functions import BinaryClassification
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv ("../input/health-insurance-cross-sell-prediction/test.csv")
#top 5 rows
train.head()

test.head()
#top 5 Rows
train.tail()

test.tail()
#no of missing values in training set
train.isnull().sum()
#Data Type of train Set
train.dtypes
#no of rows and columns in train set
train.shape
#no of rows and columns in test set
test.shape
cat = train.select_dtypes("object")

cat.dtypes
cat_data = cat.astype('category')
cat_data.dtypes
cat1 = test.select_dtypes("object")

cat1.dtypes
cat1_data = cat1.astype('category')
cat1_data.dtypes
num_data = train.select_dtypes('int64')

float_data = train.select_dtypes('float64')
num_data = test.select_dtypes('int64')

float_data = test.select_dtypes('float64')
train.describe()
test.describe()
corr_df = train.corr()

corr_df
corr_df1 = test.corr()

corr_df1
sns.heatmap(corr_df,annot = True)

sns.heatmap(corr_df1 , annot =True)
sns.countplot("Gender",data =train)

plt.show()
sns.countplot("Gender",data=test)

plt.show()
plt.style.use("fivethirtyeight")

sns.countplot(x = "Gender",data = train,hue = "Response" )

plt.show()
plt.style.use("fivethirtyeight")

sns.countplot(x = "Gender" , data =test ,hue = "Vehicle_Damage")

plt.show()
facet  =  sns.FacetGrid(train , hue ="Response" , aspect = 4)

facet.map(sns.kdeplot ,"Age" ,shade =True)

facet.set(xlim = (0,train["Age"].max()))

facet.add_legend()

plt.show()
facet  =  sns.FacetGrid(test , hue ="Vehicle_Damage" , aspect = 4)

facet.map(sns.kdeplot ,"Age" ,shade =True)

facet.set(xlim = (0,train["Age"].max()))

facet.add_legend()

plt.show()
sns.countplot("Driving_License" , data =train ,hue = "Response")

plt.show()
sns.countplot ("Age" , data = test ,hue = "Vehicle_Damage")

plt.show()
train["Response"].value_counts()
test["Vehicle_Damage"].value_counts()
sns.countplot(x = "Previously_Insured" , data =train , hue = "Response")

plt.show()
sns.countplot (x ="Previously_Insured", data =test , hue ="Vehicle_Damage")

plt.show()
train.drop(["id","Region_Code","Policy_Sales_Channel"],axis =1 ,inplace = True)

train.head()
test.drop(["id","Policy_Sales_Channel","Vintage"],axis = 1 ,inplace = True)

test.head()
le = LabelEncoder()

train["Vehicle_Damage"] = le.fit_transform(train["Vehicle_Damage"])
le1 = LabelEncoder()

test["Vehicle_Damage"] = le1.fit_transform(test["Vehicle_Damage"])
train =  pd.get_dummies(train,drop_first=True)

train.head()
test =  pd.get_dummies(test,drop_first=True)

test.head()
x = train.drop("Response", axis = 1)

y = train["Response"]
x1 = test.drop("Gender_Male", axis = 1)

y1 = test["Gender_Male"]
model = ExtraTreesClassifier()

model.fit(x,y)
plt.figure(figsize=(8,6))

important_features = pd.Series(model.feature_importances_,index = x.columns)

important_features.nlargest(7).plot(kind = "bar")

plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
dt_clf = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 100 ,random_state =0)

dt_clf.fit(x_train,y_train)
dt_pred = dt_clf.predict(x_test)

dt_accuracy = accuracy_score(y_test,dt_pred)

dt_accuracy
rf_clf = RandomForestClassifier(n_estimators = 200 ,random_state =0)

rf_clf.fit(x_train , y_train)
rf_pred = rf_clf.predict(x_test)

rf_accuracy = accuracy_score(y_test,rf_pred)

rf_accuracy
lr_clf = LogisticRegression(random_state = 0)

lr_clf.fit(x_train ,y_train)
lr_pred = lr_clf.predict(x_test)

lr_accuracy = accuracy_score(y_test,lr_pred)

lr_accuracy
knn_clf  =  KNeighborsClassifier(n_neighbors = 20)

knn_clf.fit(x_train,y_train)
knn_pred = knn_clf.predict(x_test)

knn_accuracy = accuracy_score(y_test,knn_pred)

knn_accuracy
print(classification_report(y_test,dt_pred))
print(classification_report(y_test,rf_pred))
print(classification_report(y_test,lr_pred))
print(classification_report(y_test,knn_pred))