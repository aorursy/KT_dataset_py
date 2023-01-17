import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(8,6)})
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pd.set_option('max_colwidth', 256)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
df.head()
df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df['Churn'].replace(to_replace='No',  value=0, inplace=True)
df
y = df['Churn']
df.isnull().sum()
sns.pairplot(df)
sns.barplot(x="SeniorCitizen",y="Churn",hue="gender",data=df)
sns.barplot(x="SeniorCitizen",y="Churn",hue="PhoneService",data=df)
sns.barplot(x="SeniorCitizen",y="Churn",hue="InternetService",data=df)
plt.legend(loc='center left', bbox_to_anchor=(1.00, 0.5), ncol=1)
sns.barplot(x="SeniorCitizen",y="Churn",hue="OnlineSecurity",data=df)
plt.legend(loc='upper left')
sns.barplot(x="SeniorCitizen",y="Churn",hue="DeviceProtection",data=df)
print("most important features relative to target")
corr = df.corr()
corr.sort_values(['Churn'], ascending=False, inplace=True)
corr.Churn
sns.heatmap(corr,annot=True)
sns.violinplot(y="tenure",x="gender",hue="Churn",data=df)
sns.boxplot(y="MonthlyCharges",x="gender",hue="Churn",data=df,width=0.6)
plt.legend(loc='center left', bbox_to_anchor=(1.00, 0.5), ncol=1)
df['Partner'].replace(to_replace='Yes', value=1, inplace=True)
df['Partner'].replace(to_replace='No',  value=0, inplace=True)
df['Partner'] = pd.to_numeric(df['Partner'])
df['Dependents'].replace(to_replace='Yes', value=1, inplace=True)
df['Dependents'].replace(to_replace='No',  value=0, inplace=True)
df['Dependents'] = pd.to_numeric(df['Dependents'])
df['PhoneService'].replace(to_replace='Yes', value=1, inplace=True)
df['PhoneService'].replace(to_replace='No',  value=0, inplace=True)
df['PhoneService'] = pd.to_numeric(df['PhoneService'])
df['OnlineSecurity'].value_counts()
df['PaperlessBilling'].replace(to_replace='Yes', value=1, inplace=True)
df['PaperlessBilling'].replace(to_replace='No',  value=0, inplace=True)
df['PaperlessBilling'] = pd.to_numeric(df['PaperlessBilling'])
df.head()
df1 = df.copy()
df1
df1.drop(columns='customerID',inplace=True)
df1['MonthlyCharges'] = pd.to_numeric(df1['MonthlyCharges'])
df1.drop(columns='TotalCharges',inplace=True)
cols = df1.columns
num_cols = df1._get_numeric_data().columns
num_cols
col_list = list(set(cols) - set(num_cols))
col_list
df1 = pd.get_dummies(df1, columns = ['TechSupport',
 'OnlineSecurity',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'DeviceProtection',
 'PaymentMethod',
 'InternetService',
 'OnlineBackup',
 'MultipleLines',
 'gender'],drop_first = True)
df1.head()
X = df1.iloc[:].values
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_kpred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_kpred)
print(cm)
accuracy_score(y_test, y_kpred)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_rpred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_rpred)
print(cm)
accuracy_score(y_test, y_rpred)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_spred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_spred)
print(cm)
accuracy_score(y_test, y_spred)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_slpred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_slpred)
print(cm)
accuracy_score(y_test, y_slpred)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_dpred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_dpred)
print(cm)
accuracy_score(y_test, y_dpred)