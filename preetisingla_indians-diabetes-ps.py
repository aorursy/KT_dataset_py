import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.shape
df.describe()
df.isnull().count()
df['Outcome'].value_counts()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
df.columns
sns.distplot(df['Age'], bins = 50,kde =False)
sns.countplot(x = 'Outcome', data = df)
df.hist(figsize = (12,8))
corr = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(corr, annot = True)
df.plot(kind = 'box', figsize = (12,8), subplots = True, layout = (3,3))
plt.show()
cols = df.columns[:8]
for item in cols:
    plt.figure(figsize = (6,4))
    plt.title(str(item) + 'With' + 'Outcome')
    sns.violinplot(x = df.Outcome, y = df[item], data = df)
    plt.show()
#sns.pairplot(df,hue='Outcome',palette='coolwarm', diag_kind = 'hist')
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predic_logistic = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predic_logistic))
print(confusion_matrix(y_test, predic_logistic))
print('Accuracy -- >', logmodel.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, predic_logistic)
sns.heatmap(cm, annot = True, fmt = 'g')
plt.show()
from sklearn.ensemble import RandomForestClassifier
model_random = RandomForestClassifier()
model_random.fit(X_train, y_train)
y_predict_random = model_random.predict(X_test)
print(classification_report(y_test, y_predict_random))
print(confusion_matrix(y_test, y_predict_random))
print('Accuracy -- >', model_random.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_predict_random)
sns.heatmap(cm, annot = True, fmt = 'g')
plt.show()
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
print('Accuracy -- >', model_svc.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred_svc)
sns.heatmap(cm, annot = True, fmt = 'g')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)
y_pred_knn = model_KNN.predict(X_test)
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print('Accuracy -- >', model_KNN.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot = True, fmt = 'g')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
model_GB = GradientBoostingClassifier()
model_GB.fit(X_train, y_train)
y_pred_GB = model_GB.predict(X_test)
print(classification_report(y_test, y_pred_GB))
print(confusion_matrix(y_test, y_pred_GB))
print('Accuracy -- >', model_GB.score(X_test, y_test)*100)
cm = confusion_matrix(y_test, y_pred_GB)
sns.heatmap(cm, annot = True, fmt = 'g')
plt.show()