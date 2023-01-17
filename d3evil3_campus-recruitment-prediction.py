import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.describe()
df.info()
df.isna().sum()
sns.countplot(df['gender'])
sns.heatmap(df.corr(),annot=True,fmt = '.0%')
data_mean = df
hist_mean = data_mean.hist(bins=10,grid=False, figsize=(15, 10))


plt = data_mean.plot(kind= 'density', subplots=True, layout=(4,3), sharex=False, 

                     sharey=False,fontsize=12, figsize=(15,10))
from sklearn.preprocessing import LabelEncoder

object_cols= ['gender','hsc_s','degree_t','workex','specialisation','status']



label_encoder = LabelEncoder()

for col in object_cols:

    df[col]= label_encoder.fit_transform(df[col])

df.head()
df.columns
X = df[['gender', 'ssc_p','hsc_p', 'hsc_s',

       'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']]

y= df['status']
df.head()

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
#logistic Regression 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

print("Logistic Regression accuracy : {:.2f}%".format(model.score(X_test,y_test)*100))
# Support Vactor 

from sklearn.svm import SVC

svm = SVC(random_state=1)

svm1 = SVC(kernel='linear',gamma='scale',random_state=0)

svm2 = SVC(kernel='rbf',gamma='scale',random_state=0)

svm3 = SVC(kernel='poly',gamma='scale',random_state=0)

svm4 = SVC(kernel='sigmoid',gamma='scale',random_state=0)



svm.fit(X_train,y_train)

svm1.fit(X_train,y_train)

svm2.fit(X_train,y_train)

svm3.fit(X_train,y_train)

svm4.fit(X_train,y_train)



print('SVC Accuracy : {:,.2f}%'.format(svm.score(X_test,y_test)*100))

print('SVC Liner Accuracy : {:,.2f}%'.format(svm1.score(X_test,y_test)*100))

print('SVC RBF Accuracy : {:,.2f}%'.format(svm2.score(X_test,y_test)*100))

print('SVC Ploy Accuracy : {:,.2f}%'.format(svm3.score(X_test,y_test)*100))

print('SVC Sigmoid Accuracy : {:,.2f}%'.format(svm4.score(X_test,y_test)*100))
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

print(" Naive Bayes accuracy : {:.2f}%".format(nb.score(X_test,y_test)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000,random_state=1)

rf.fit(X_train,y_train)

print("Random Forest Classifier accuracy : {:.2f}%".format(rf.score(X_test,y_test)*100))
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=0)

dt.fit(X_train,y_train)

print("Decision Tree Accuracy : {:,.2f}%".format(dt.score(X_test,y_test)*100))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000,random_state=1)

rf.fit(X_train,y_train)

print("Random Forest Regressor accuracy : {:.2f}%".format(rf.score(X_test,y_test)*100))
# Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(criterion='mse',max_depth=4, random_state=0)

dt.fit(X_train,y_train)

print("Decision Tree Accuracy : {:,.2f}%".format(dt.score(X_test,y_test)*100))
import xgboost

xg = xgboost.XGBClassifier()

xg.fit(X_train,y_train)

print("XGboost accuracy : {:.2f}%".format(xg.score(X_test,y_test)*100))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

print("K-Neighbors Accuracy : {:,.2f}%".format(knn.score(X_test,y_test)*100))