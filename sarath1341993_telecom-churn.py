import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,roc_auc_score
data=pd.read_csv("../input/bigml_59c28831336c6604c800002a.csv")
print(data.shape)
print(data.columns)
data.head()
data.describe()
data.dtypes
data.isnull().sum()
data.drop(["phone number"],axis=1,inplace=True)
data["account length"].value_counts().head(20)
for i in data.columns:
    if data[i].dtype == "object":
        print(data[i].value_counts())
ax=sns.countplot(x="churn",data=data)
for p in ax.patches:
        ax.annotate('{:.1f}%'.format( (p.get_height()/data.shape[0])*100 ), (p.get_x()+0.3, p.get_height()))
# data.groupby(["area code","churn"]).size()
ac=data.groupby(["area code", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
for i in ac.patches:
    ac.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))
# data["state"].value_counts()
st=data.groupby(["state", "churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(15,5))
# for i in st.patches:
#     st.text(i.get_x(), i.get_height(),str(i.get_height()))
# cols=['state','area code',
#  'international plan',
#  'voice mail plan',
#  'number vmail messages','customer service calls',]
# plt.plot([1,6])
# for i in range(len(cols)):
#     plt.subplot(i+1,1,1)
#     a=data.groupby([cols[i], "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
#     for i in a.patches:
#         a.text(i.get_x(), i.get_height(),str(i.get_height()))
#     plt.show()
ip=data.groupby(["international plan", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
for i in ip.patches:
    ip.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))
vp=data.groupby(["voice mail plan", "churn"]).size().unstack().plot(kind='bar', stacked=True,figsize=(6,5))
# for i in vp.patches:
#     vp.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))
cs=data.groupby(["customer service calls", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(12,6))
for i in cs.patches:
    cs.text(i.get_x()+0.05, i.get_height()+20,int(i.get_height()))
cate = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['bool', 'object']]
le = preprocessing.LabelEncoder()
for i in cate:
    le.fit(data[i])
    data[i] = le.transform(data[i])
data.head()
y=data["churn"]
x=data.drop(["churn"],axis=1)
x.columns
clf = RandomForestClassifier()
clf.fit(x, y)
clf.score(x,y)
clf.feature_importances_
importances = clf.feature_importances_
indices = np.argsort(importances)
features=x.columns
fig, ax = plt.subplots(figsize=(9,9))
plt.title("Feature Impoprtance")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices] )
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf.fit(X_train,y_train)

print("Train accuracy: ",clf.score(X_train,y_train))

print("Test accuracy: ",clf.score(X_test,y_test))
from sklearn import tree

dt=tree.DecisionTreeClassifier()
dt

dt.fit(X_train,y_train)
print("Train data accuracy:",dt.score(X_train,y_train))

print("Test data accuracy:",dt.score(X_test,y_test))
data=pd.read_csv("../input/bigml_59c28831336c6604c800002a.csv")
data.head()
data.drop(["phone number"],axis=1,inplace =True)
data.dtypes
data.dtypes.value_counts()
cate
enc=pd.get_dummies(data[cate[:-1]])
enc.columns
data.columns
data.drop(cate[:-1],axis=1,inplace=True)
data[enc.columns]=enc
data.shape
X=data.drop(["churn"],axis=1)
y=data["churn"]
X.shape
lr=LogisticRegression().fit(X, y)
lr.score(X, y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
lr=LogisticRegression().fit(X_train, y_train)
print("Train accuracy:",lr.score(X_train, y_train))
print("Test accuracy:",lr.score(X_test,y_test))
X_train.columns
joblib.dump(X_train,'X_train.pkl') 
joblib.dump(y_train,'y_train.pkl') 
joblib.dump(X_test,'X_test.pkl') 
joblib.dump(y_test,'y_test.pkl')
joblib.dump(x,'x.pkl')
joblib.dump(y,'y.pkl')
X_train=joblib.load('X_train.pkl')
y_train=joblib.load('y_train.pkl')
X_test=joblib.load('X_test.pkl')
y_test=joblib.load('y_test.pkl')
x=joblib.load('x.pkl')
y=joblib.load('y.pkl')
algo = pd.DataFrame(columns=["Algorithm","Accuracy","auc score"])
algo.head()
clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = MultinomialNB()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)],
              ["Algorithm","Accuracy","auc score"])
algo=algo.append([lr],ignore_index=True)
algo.sort_values(["Accuracy"], ascending=[False])