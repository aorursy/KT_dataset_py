import pandas as pd

import tensorflow

import numpy

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("../input/server-logs-suspicious/CIDDS-001-external-week1.csv")
df.head()
df.columns
df.isnull().sum()
df.dtypes
df["class"].unique()
data = df.copy()

data.shape

sns.set(style = "darkgrid")

sns.countplot(x = "class",data=data)
df["Flags"].unique()
df["A"]=0

df["P"]=0

df["S"]=0

df["R"]=0

df["F"]=0

df["x"]=0
def set_flag(data,check):

    val=0;

    if(check in list(data["Flags"])):

        val = 1 ;

    return val;
df.columns
df["A"] = df.apply(set_flag,check ="A", axis = 1)

df["P"] = df.apply(set_flag,check = "P" ,axis = 1)

df["S"] = df.apply(set_flag,check ="S",axis = 1)

df["R"] = df.apply(set_flag,check="R" ,axis = 1)

df["F"] = df.apply(set_flag,check ="F" ,axis = 1)

df["x"] = df.apply(set_flag,check ="x" ,axis = 1)
sns.countplot(x="S",hue = "class",data=df)
sns.countplot(x = "Proto",hue = "class",data = df)
df=df.drop(columns = ["Date first seen","attackType","attackID","attackDescription","Flows","Tos","Flags"])
df.head()
import re

def convtonum(data):

    num1=data["Bytes"]

    if "M" in data["Bytes"]:

        num=re.findall("[0-9.0-9]",data["Bytes"])

        num1 = float("".join(num))*100000

    num1 = float(num1)

    return num1
df["Bytes"] = df.apply(convtonum,axis = 1)
df.head()
from sklearn.preprocessing import LabelEncoder

col = ["Proto","class","Src IP Addr","Dst IP Addr"]

enc = LabelEncoder()

for col_name in col:

    df[col_name]=enc.fit_transform(df[col_name])

data1 = df.copy()

plt.figure(figsize=(18,5))

sns.heatmap(data1.corr(),annot=True,cmap = "RdYlGn")
data_y = data1["class"]

data_x = data1.drop(columns = ["class"])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
#decision-tree-classifier - single-tree-classifier  // using all features



from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(criterion="entropy", max_depth=10) # you can use GINI index also here as a critirion 

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(15).plot(kind='barh')

plt.show()
# if you want to select most important features from an algorithm use recursive feature elimination and run algorithm on that



from sklearn.feature_selection import RFE



m = DecisionTreeClassifier(criterion="entropy", max_depth=10)

rfe = RFE(m,8)

fit=rfe.fit(X_train,y_train)



print(X_train.columns)

print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)





