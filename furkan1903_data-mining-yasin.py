# Import libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 50)
# Load the dataset

df = pd.read_csv('../input/winequalityN.csv')

df.head()
df.info()
df.shape
df.dropna(inplace=True)
df['quality'].unique()
corr=df.corr()
plt.figure(figsize=(14,6))

sns.heatmap(corr,annot=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['type'] = le.fit_transform(df['type'])
le.classes_
le.transform(le.classes_)
dict(zip(le.classes_, le.transform(le.classes_)))
df.head()
df['type'].value_counts()
# {'red': 0, 'white': 1}







plt.figure(figsize=(15,7))

 

# Data to plot

labels = 'white', 'red'

sizes = [4870,1593]

colors = ['white', 'red']

explode = (0.1, 0 )  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('The percentage of type of wine',fontsize=20)

plt.legend(('white', 'red'),fontsize=15)

plt.axis('equal')

plt.show()
corr=df.corr()

plt.figure(figsize=(14,6))

sns.heatmap(corr,annot=True)
# i choose 'total sulfur dioxide' because it has 0.7 corrolation with type 

# and 'free sulfur dioxide' because it has 0.47 corrolation with type

X = df[['free sulfur dioxide', 'total sulfur dioxide']]

y = df['type']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test  = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
print(' Logistic Regression Accuracy = ',round(accuracy_score(y_test,y_pred),4) *100, '%')
# Making the Confusion Matrix will contain the correct and incorrect prediction on the dataset.

from sklearn.metrics import confusion_matrix



cm_log_reg = confusion_matrix(y_test, y_pred)

print(cm_log_reg)
X_train
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)

print('KNN Accuracy = ',round(accuracy_score(y_test,knn_pred),4) *100, '%')
# Making the Confusion Matrix will contain the correct and incorrect prediction on the dataset.

from sklearn.metrics import confusion_matrix



cm_knn = confusion_matrix(y_test, knn_pred)

print(cm_knn)
from sklearn.svm import SVC

svm_linear=SVC(kernel='linear').fit(X_train,y_train)

svm_pred=svm_linear.predict(X_test)

print('SVM Accuracy = ',round(accuracy_score(y_test,svm_pred),4) *100, '%')
# Making the Confusion Matrix will contain the correct and incorrect prediction on the dataset.

cm_svm_lin = confusion_matrix(y_test, svm_pred)

print(cm_svm_lin)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB().fit(X_train,y_train)

nb_pred=nb.predict(X_test)

print('Naive bayes Accuracy = ',round(accuracy_score(y_test,nb_pred),4) *100, '%')
