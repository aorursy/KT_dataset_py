# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
%matplotlib inline
style.use('fivethirtyeight')
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()
df.duplicated().sum()
plt.figure(figsize=(12,10))
sns.countplot(df.Pregnancies,hue=df['Outcome'])
plt.figure(figsize=(12,10))
sns.countplot(df.Age,palette='rainbow')
plt.tight_layout()
df['age_range'] = [np.floor(i/10)*10 for i in df['Age'] ]
display(df.head())
df1 = df.groupby('age_range',as_index=False)
df1.agg({'Outcome':'mean'})
df.drop('age_range',axis=1,inplace=True)
df.head()
plt.figure(figsize=(10,10))
plt.pie(df.Outcome.value_counts(),autopct='%0.2f%%',labels=[0,1])
plt.show()
g = sns.FacetGrid(df,hue='Outcome',palette='Set1',height=6,aspect=2)
g.map(plt.hist,'Glucose',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='coolwarm',height=6,aspect=2)
g.map(plt.hist,'BloodPressure',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='Set2',height=6,aspect=2)
g.map(plt.hist,'SkinThickness',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='dark',height=6,aspect=2)
g.map(plt.hist,'Insulin',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='rainbow',height=6,aspect=2)
g.map(plt.hist,'BMI',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='cool',height=6,aspect=2)
g.map(plt.hist,'DiabetesPedigreeFunction',alpha=0.6,bins=20)
plt.legend()
g = sns.FacetGrid(df,hue='Outcome',palette='hot',height=6,aspect=2)
g.map(plt.hist,'Age',alpha=0.6,bins=20)
plt.legend()
x1 = np.round(df[df['Outcome'] == 1][['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].mean())
x2 = np.round(df[df['Outcome'] == 0][['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].mean())
display(x1)
display(x2)
df.loc[df['Outcome'] == 1,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df.loc[df['Outcome'] == 1,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,x1)
df.loc[df['Outcome'] == 0,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df.loc[df['Outcome'] == 0,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,x2)
df.head()
sns.pairplot(df,hue='Outcome',diag_kind='hist')
df.describe()
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap='BuGn')
plt.tight_layout()
plt.figure(figsize=(20,20))

ax1 = plt.subplot(4,2,1)
sns.violinplot(x=df['Outcome'],y=df['Pregnancies'],palette='cool')

ax2 = plt.subplot(4,2,2)
sns.violinplot(x=df['Outcome'],y=df['Glucose'],palette='coolwarm')

ax3 = plt.subplot(4,2,3)
sns.violinplot(x=df['Outcome'],y=df['BloodPressure'],palette='hot')

ax4 = plt.subplot(4,2,4)
sns.violinplot(x=df['Outcome'],y=df['SkinThickness'],palette='dark')

ax5 = plt.subplot(4,2,5)
sns.violinplot(x=df['Outcome'],y=df['Insulin'],palette='Set1')

ax6 = plt.subplot(4,2,6)
sns.violinplot(x=df['Outcome'],y=df['BMI'],palette='Set2')

ax7 = plt.subplot(4,2,7)
sns.violinplot(x=df['Outcome'],y=df['DiabetesPedigreeFunction'],palette='rainbow')

ax8 = plt.subplot(4,2,8)
sns.violinplot(x=df['Outcome'],y=df['Age'])

plt.tight_layout()
plt.show()
#from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler()
#scalar.fit(df.drop('Outcome',axis=1))
#scaled_feat = scalar.transform(df.drop('Outcome',axis=1))
#scaled_feat = pd.DataFrame(scaled_feat,columns=df.columns[0:-1])
#scaled_feat.head()
#X = scaled_feat
X  = df.drop('Outcome',axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
from sklearn.metrics import accuracy_score
print("The Accuracy of the model with K=3 equal {} %".format(accuracy_score(y_test, pred)*100))
error_rate = []
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != y_test))
plt.figure(figsize=(10,8))
plt.plot(range(len(error_rate)),error_rate,marker='o',markerfacecolor='white',linestyle='--')
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)
print(confusion_matrix(y_test,knn_predict))
print('\n')
print(classification_report(y_test,knn_predict))
print("The Accuracy of the model with K=12 equal {} %".format(accuracy_score(y_test, knn_predict)*100))
plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,knn_predict)),annot=True,cmap='Dark2',cbar=False,linewidths=1,fmt='.3g')
error_rate = []
for i in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=12,p=i)
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != y_test))
plt.figure(figsize=(10,8))
plt.plot(range(len(error_rate)),error_rate,marker='o',markerfacecolor='white',linestyle='--',color='orange')
knn = KNeighborsClassifier(n_neighbors=12,p=1)
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)
print(confusion_matrix(y_test,knn_predict))
print('\n')
print(classification_report(y_test,knn_predict))
print("The Accuracy of the model with K=12 and p = 1 equal {} %".format(accuracy_score(y_test, knn_predict)*100))
plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,knn_predict)),annot=True,cmap='Set3',cbar=False,linewidths=1,fmt='.3g')