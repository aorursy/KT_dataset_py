#importing necesary packages
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head(10)
data.info()
#quality directly does not depend on fixed acity
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='fixed acidity', data=data, palette="magma")
#as quality increases voaltile acidity decreases
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='volatile acidity', data=data, palette="magma")
#as quality increases citric acid content increases
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='citric acid', data=data, palette="magma")
#as quality does not directly depend on residual sugar content
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='residual sugar', data=data, palette="magma")
#as quality increases chlorides content decreases
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='chlorides', data=data, palette="magma")
#medium quality red wine have highest free sulfur dioxide
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='free sulfur dioxide', data=data, palette="magma")
#medium quality red wine have highest total sulfur dioxide
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='total sulfur dioxide', data=data, palette="magma")
#quality directly depends on sulphates
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='sulphates', data=data, palette="magma")
#quality directly depends on alcohol
plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='alcohol', data=data, palette="magma")
#density and pH is almost similar in all cases hence we avoid it
print("Respectively Maximum and Minimum value of density: ", data['density'].max(), "\t", data['density'].min())
print("Respectively Maximum and Minimum value of pH: ", data['pH'].max(), "\t", data['pH'].min())
#Dividing wine as good and bad by giving the limit for the quality
bins=(2, 6.5, 8)
group_names=['bad', 'good']
data['quality']=pd.cut(data['quality'], bins=bins, labels=group_names)
#changing the good labels to 1 and bad to 0 with a label encoder
enc=LabelEncoder()
data['quality']=enc.fit_transform(data['quality'])
data['quality'].value_counts()
X=data.drop('quality', axis=1)
y=data['quality']
sc=StandardScaler()
X=sc.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
lr=LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr=lr.predict(X_test)
accuracy_score(y_test, y_pred_lr)*100
print(classification_report(y_test, y_pred_lr))
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf=rf.predict(X_test)
accuracy_score(y_test, y_pred_rf)*100
print(classification_report(y_test, y_pred_rf))
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)
accuracy_score(y_test, y_pred_dt)*100
print(classification_report(y_test, y_pred_dt))
svc=SVC()
svc.fit(X_train, y_train)
y_pred_svm=svc.predict(X_test)
accuracy_score(y_test, y_pred_svm)*100
print(classification_report(y_test, y_pred_svm))