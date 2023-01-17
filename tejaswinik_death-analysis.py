import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
data=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
col_list=list(data.columns)
cnt=data.DEATH_EVENT.value_counts()

plt.bar(['0','1'],cnt)
X=data.drop('DEATH_EVENT',axis=1)

y=data['DEATH_EVENT']
scale=StandardScaler()

scaled_data=pd.DataFrame(scale.fit_transform(X),columns=col_list[:-1])
cor=scaled_data.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(cor,vmin=-1,vmax=1,cmap='RdYlGn',annot=True)
X_train,X_test,y_train,y_test=train_test_split(scaled_data,y,test_size=0.2)
logi=LogisticRegression()

logi.fit(X_train,y_train)

y_pred=logi.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)
decision_tree=DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)

y_pred=decision_tree.predict(X_test)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
r_forest=RandomForestClassifier()

r_forest.fit(X_train,y_train)

y_pred=r_forest.predict(X_test)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
class_0,class_1=data.DEATH_EVENT.value_counts()

df_class_0=data[data['DEATH_EVENT']==0]

df_class_1=data[data['DEATH_EVENT']==1]
df_class_1 =df_class_1.sample(class_0,replace=True)

df_class_1=df_class_1.reset_index(drop=True)

resampled_data=pd.concat([df_class_0,df_class_1],axis=0).reset_index()
cnt=resampled_data.DEATH_EVENT.value_counts()

plt.bar(['0','1'],cnt)
X=resampled_data.drop(['DEATH_EVENT','index'],axis=1)

y=resampled_data['DEATH_EVENT']
scale=StandardScaler()

scaled_data=pd.DataFrame(scale.fit_transform(X),columns=col_list[:-1])
X_train,X_test,y_train,y_test=train_test_split(scaled_data,y,test_size=0.2)
logi=LogisticRegression()

logi.fit(X_train,y_train)

y_pred=logi.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)
decision_tree=DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)

y_pred=decision_tree.predict(X_test)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
r_forest=RandomForestClassifier()

r_forest.fit(X_train,y_train)

y_pred=r_forest.predict(X_test)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)