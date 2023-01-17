import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import seaborn as sns  
from sklearn.feature_selection import SelectKBest, chi2
# import matplotlib.pyplot as plt
# sns.set(style="ticks", color_codes=True)
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

df.head()
df.shape
df.drop('Unnamed: 32',axis=1,inplace=True)
df.drop('id',axis=1,inplace=True)
df.columns
df.describe()
df['diagnosis'].value_counts().plot(kind='bar')
a=sns.countplot(df['diagnosis'])
labelencoder = LabelEncoder()
df['encoded_labels'] = labelencoder.fit_transform(df['diagnosis'])
df
df.drop('diagnosis',axis=1,inplace=True)

Y=df.encoded_labels
Y
df.drop('encoded_labels',axis=1,inplace=True)
df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
best_features=SelectKBest(score_func=chi2,k=20)
best_features_fit=best_features.fit(x_train,y_train)
best_features_fit
best_features_fit.scores_
x_train_2 = best_features_fit.transform(x_train)
x_test_2 = best_features_fit.transform(x_test)
x_train_2
x_train_2.shape
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")
df=df.iloc[0]
type(df)
clf_rf_3 = RandomForestClassifier()      
from sklearn.model_selection import KFold 
kf = KFold(n_splits=5) 
kf.get_n_splits(df)
print(kf)
# mat=[]
# for train_index, test_index in kf.split(df):
#     print("TRAIN",train_index,"TEST",test_index)
#     ky_train,ky_test=Y[train_index],Y[test_index]
#     kX_train,kX_test=df[train_index],df[test_index]
#     clr_rf_3 = clf_rf_3.fit(kX_train,ky_train)
#     ac = accuracy_score(ky_test,clf_rf_3.predict(kX_test))
#     mat.append(ac)
# mat
