import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv(r"D:\data\customer_segmentation_AV\Train_aBjfeNk.csv")
data.head()
data.shape
data.info()
data["Gender"].value_counts()
data["Graduated"].value_counts()
data["Profession"].value_counts()
data["Work_Experience"].value_counts()
data["Spending_Score"].value_counts()
data["Family_Size"].value_counts()
data["Var_1"].value_counts()
data["Segmentation"].value_counts()
data["Segmentation"].value_counts().plot(kind='bar')
data["Age"].hist()
sns.catplot(x="Segmentation", y="Age", data=data);
pd.pivot_table(data,values="Age",columns="Segmentation",index="Var_1",aggfunc="count")
data.isnull().sum()
data=data.drop("ID",axis=1)
data["Ever_Married"]=data["Ever_Married"].fillna('')
data["Graduated"]=data["Graduated"].fillna('')
data["Profession"]=data["Profession"].fillna('')
data["Var_1"]=data["Var_1"].fillna('')

data["Work_Experience"]=data["Work_Experience"].fillna(data["Work_Experience"].mean())
data["Family_Size"]=data["Family_Size"].fillna(data["Family_Size"].mean())
data.isnull().sum()
#data=pd.get_dummies(data)
x=data.drop("Segmentation",axis=1)

y=data["Segmentation"]
x=pd.get_dummies(x)
x.shape,y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=45)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score,roc_auc_score
rf = RandomForestClassifier(n_estimators=150,criterion='gini',min_samples_leaf=3,class_weight="balanced")
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(X=x_test)
y_pred_rf
print(accuracy_score(y_test,y_pred_rf))
#print(roc_auc_score(y_test,y_pred_rf))
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(X=x_test)
print(accuracy_score(y_test,y_pred_dt))
from sklearn.ensemble import BaggingClassifier
bag_cf=BaggingClassifier(dt,n_estimators=100, random_state=10,max_features=3)
bag_cf.fit(x_train,y_train)
y_pred_bag=bag_cf.predict(x_test)
accuracy_score(y_test,y_pred_bag)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=20)
kn.fit(x_train,y_train)
y_pred_kn=kn.predict(x_test)
accuracy_score(y_test,y_pred_kn)
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=200,learning_rate=.1,max_depth=3,random_state=42)
gbc.fit(x_train,y_train)
y_pred_gbc=gbc.predict(x_test)
accuracy_score(y_test,y_pred_gbc)
train=pd.read_csv(r"D:\data\customer_segmentation_AV\Test_LqhgPWU.csv")
train.head()

train.shape
train.isnull().sum()
train=train.drop("ID",axis=1)
train["Graduated"]=train["Graduated"].fillna('')
train["Ever_Married"]=train["Ever_Married"].fillna('')
train["Profession"]=train["Profession"].fillna('')
train["Var_1"]=train["Var_1"].fillna('')
train["Family_Size"]=train["Family_Size"].fillna(train["Family_Size"].mean())
train["Work_Experience"]=train["Work_Experience"].fillna(train["Work_Experience"].mean())
train=pd.get_dummies(train)
train.shape
y_test_pred=gbc.predict(train)
y_test_pred
train["Segmentation"]=y_test_pred
train.head()
submision=train[["ID","Segmentation"]]
submision.to_csv(r"D:\data\customer_segmentation_AV\submission.csv")


from sklearn.ensemble import AdaBoostClassifier
adb=AdaBoostClassifier(n_estimators=200,learning_rate=1,random_state=42)
adb.fit(x_train,y_train)
y_pred_adb=adb.predict(x_test)
accuracy_score(y_test,y_pred_adb)
adb_cm = confusion_matrix(y_test,y_pred_adb)
sns.heatmap(adb_cm, annot=True, fmt='d',xticklabels = ["A", "B","C","D"] , yticklabels = ["A", "B","C","D"],cmap='Set2')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('CM_AdaBoosting')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='sgd',activation='relu',hidden_layer_sizes=(100,20))
clf.fit(x_train,y_train)
y_pred_mlp=clf.predict(x_test)
accuracy_score(y_test,y_pred_mlp)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X=x_train, y=y_train)
y_pred_lr=lr_model.predict(X=x_test)
print(accuracy_score(y_test,y_pred_lr))

