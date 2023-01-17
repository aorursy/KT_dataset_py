# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/indian_liver_patient.csv')
df.head()
df.shape
df.isnull().sum()
df['Albumin_and_Globulin_Ratio'].value_counts()
df.describe().T
#here we see one of our feature contains missing values so we replace it with mode 
df['Albumin_and_Globulin_Ratio']=df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mode()[0])
df.isnull().sum()
df=df.rename(columns={
                      'Age':'Age',
                      'Gender':'Gender',
                      'Total_Bilirubin':'TBilirubin',
                      'Direct_Bilirubin':'DirectBilirubin',
                      'Alkaline_Phosphotase':'AlkalinePhosphotase',
                      'Alamine_Aminotransferase':'AlamineAminotransferase',
                      'Aspartate_Aminotransferase':'AspartateAminotransferase',
                      'Total_Protiens':'Proteins',
                      'Albumin':'Albumin',
                      'Albumin_and_Globulin_Ratio':'AlbuminGlobulin',
                      'Dataset':'Label'
    
})
df=df.replace({
    'Label':2
},0)
df.head(10)
df["Gender"]=df["Gender"].astype('category').cat.codes
front=df['Label']
df.drop('Label',axis=1,inplace=True)
df.insert(0,'Label',front)
df.head()
corr=df.corr(method='spearman')
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
corr
#here we see mens having more liver diseases as compared to women.
cross_table=pd.crosstab(index=df['Gender'],columns=df['Label'])
cross_table.plot(kind="bar",
                figsize=(7,7),
                stacked=False)
df['Label'].value_counts()
g=sns.PairGrid(df,hue="Label",vars=['Age','Gender','Proteins','Albumin'])
g.map(plt.scatter)

sns.distplot(df['Label'],kde=False)
sns.countplot(x='Label',data=df,palette='bwr')
df.Gender.value_counts()
men_population=df[df['Gender']==1]['Label'].mean()
women_population=df[df['Gender']==0]['Label'].mean()
print(men_population*100,women_population*100)
#here we clearly see mostly people are in the range of 30 to 50
plt.hist(df['Age'],color='blue')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
train=df.drop('Label',axis=1)
test=df.Label
print(train.shape,test.shape)
train=np.array(train)
test=np.array(test)
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.25,random_state=2019)
rs_x=StandardScaler()

X_train=rs_x.fit_transform(X_train)
X_test=rs_x.fit_transform(X_test)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_train)
print(accuracy_score(y_train,y_pred))
knn1=KNeighborsClassifier(n_neighbors=7)
knn1.fit(X_train,y_train)
y_pred1=knn1.predict(X_test)
print(accuracy_score(y_test,y_pred1))
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred2=dtc.predict(X_test)
print(accuracy_score(y_test,y_pred2))
rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
y_pred3=rfc.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
g=sns.heatmap(confusion_matrix(y_test,y_pred3),annot=True,fmt="d")
#soon i will update the kernel still working on it 