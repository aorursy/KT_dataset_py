# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as py
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.offline import plot
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()

placed=df.dropna()
N_placed=df[df["status"]=="Not Placed"]
placed.head()
N_placed.head()
fig=plt.figure(figsize=(18,10))
plt.subplot(1,3,1)
plt.pie(df["gender"].value_counts(),labels={"Male"," Female"},colors={"cyan", "yellow"},
        shadow=True,autopct = '%.2f%%')
plt.title("Total Student")
plt.subplot(1,3,2)
plt.pie(placed["gender"].value_counts(),labels={"Male"," Female"},colors={"blue", "pink"},
        shadow=True,autopct = '%.2f%%')
plt.title("Placed Student")
plt.subplot(1,3,3)
plt.pie(N_placed["gender"].value_counts(),labels={"Male"," Female"},colors={"green", "orange"},
        shadow=True,autopct = '%.2f%%')
plt.title("Not Placed Student")
sns.pairplot(data=df,kind="scatter",hue="gender")
sns.pairplot(data=df,kind="scatter",hue="status")
gen=px.scatter_3d(df,x="ssc_p",y="hsc_p",z="degree_p",color="status")
iplot(gen)
gen=px.scatter_3d(df,x="mba_p",y="etest_p",z="degree_p",color="status")
iplot(gen)
fig=plt.figure(figsize=(12,6))
sns.countplot("specialisation", hue="status", data=df)
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "gender", data=df)
plt.show()
plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=df)
plt.show()
df.drop("hsc_b",inplace=True,axis=1)
df.drop("ssc_b",inplace=True,axis=1)
df.drop("sl_no",inplace=True,axis=1)
X=df.iloc[:,:-2].values
Y=df.iloc[:,-2].values
X
Y
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,5]=labelencoder_X.fit_transform(X[:,5])
X[:,6]=labelencoder_X.fit_transform(X[:,6])
X[:,8]=labelencoder_X.fit_transform(X[:,8])
Y=labelencoder_X.fit_transform(Y)
X
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y_test, Y_pred)
print(auc)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
accuracy_score(Y_test, Y_pred)
imp=classifier.feature_importances_*100
Fec=pd.DataFrame(imp,columns=["Importance"])

Nam=["Gender","SSC %","HSC %","HSC Stream","Degree % ","Degree Stream",
              "Work Ex","Entrance %"," Specialisation","Mba %"]
Fec["Features"]=Nam
Fec.head(10)
fig=plt.figure(figsize=(12,6))
sns.barplot(Fec.Features,Fec.Importance)
from sklearn.ensemble import RandomForestClassifier
#Using Random Forest Algorithm
random_forest = RandomForestClassifier(n_estimators=30,random_state=0)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y_test, Y_pred)
print(auc)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
accuracy_score(Y_test, Y_pred)
imp=random_forest.feature_importances_*100
Fec=pd.DataFrame(imp,columns=["Importance"])

Nam=["Gender","SSC %","HSC %","HSC Stream","Degree % ","Degree Stream",
              "Work Ex","Entrance %"," Specialisation","Mba %"]
Fec["Features"]=Nam
Fec.head(10)
fig=plt.figure(figsize=(12,6))
sns.barplot(Fec.Features,Fec.Importance)