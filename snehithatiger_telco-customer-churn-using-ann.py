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
#importing the dataset

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.columns
df.shape
df.info()
df.describe()
# checking if any null data exists
df.isnull().sum()
df = df.drop(columns = ['customerID'])
df.head()
#plotting the graph

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x = "Churn", data = df)
df.loc[:, 'Churn'].value_counts()
sns.countplot(x = "SeniorCitizen", data = df)
df.loc[:, 'SeniorCitizen'].value_counts()
sns.countplot(x = "InternetService", data = df)
df.loc[:, 'InternetService'].value_counts()
sns.countplot(x = "PhoneService", data = df)
df.loc[:, 'PhoneService'].value_counts()
plt.figure()
Corr=df[df.columns].corr()
sns.heatmap(Corr,annot=True)
df['TotalCharges'].value_counts().sort_index().plot.hist()
df['MonthlyCharges'].value_counts().sort_index().plot.hist()
df['tenure'].value_counts().sort_index().plot.hist()
df['PaymentMethod'].value_counts().plot.pie()
plt.gca().set_aspect('equal')
sns.kdeplot(df['tenure'], df['MonthlyCharges'])
# converting the non-numeric data into numeric data.
from sklearn.preprocessing import LabelEncoder
encoded = df.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
encoded.head()
plt.figure(figsize =(20,20))
Corr=encoded[encoded.columns].corr()
sns.heatmap(Corr,annot=True)
sns.violinplot(x='gender', y='InternetService', data=encoded)
sns.violinplot(x='PaperlessBilling', y='PaymentMethod', data=encoded)
sns.violinplot(encoded['StreamingTV'],encoded['StreamingMovies'])
sns.violinplot(encoded['Partner'],encoded['Dependents'])
sns.violinplot(encoded['MultipleLines'])
X = encoded.iloc[:, 0:19]
y = encoded.Churn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#print length of X_train, X_test, y_train, y_test
print ("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
knc.fit(X_train, y_train)
print('Accuracy score of KNN training set: {:.3f}'.format(knc.score(X_train, y_train)))
print('Accuracy score of KNN test set: {:.3f}'.format(knc.score(X_test, y_test)))
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
y_knc = knc.predict(X_test)

print('confusion_matrix of KNN: ', confusion_matrix(y_test, y_knc))
print('precision_score of KNN: ', precision_score(y_test, y_knc))
print('recall_score of KNN: ', recall_score(y_test, y_knc))
print('precision_recall_curve of KNN: ', precision_recall_curve(y_test, y_knc))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("linear model intercept (b) :{:.3f}".format(lr.intercept_))
print('linear model coeff (w) :{}'.format(lr.coef_))
print('Accuracy score of Linear Regression training set: {:.3f}'.format(lr.score(X_train, y_train)))
print('Accuracy score Linear Regression test set: {:.3f}'.format(lr.score(X_test, y_test)))
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
print('Accuracy score (training): {:.3f}'.format(rfr.score(X_train, y_train)))
print('Accuracy score (test): {:.3f}'.format(rfr.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('Accuracy score Random Forest Classifier training set: {:.3f}'.format(rfc.score(X_train, y_train)))
print('Accuracy score Random Forest Classifier test set: {:.3f}'.format(rfc.score(X_test, y_test)))
y_rfc = rfc.predict(X_test)

print('confusion_matrix of Random Forest Classifier: ', confusion_matrix(y_test, y_rfc))
print('precision_score of Random Forest Classifier: ', precision_score(y_test, y_rfc))
print('recall_score of Random Forest Classifier: ', recall_score(y_test, y_rfc))
print('precision_recall_curve of Random Forest Classifier: ', precision_recall_curve(y_test, y_rfc))
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

print('accuracy of Decision Tree Classifier training set: {:.3f}'.format(classifier.score(X_train,y_train)))
print('accuaracy of Decision Tree Classifier test set: {:.3f}'.format(classifier.score(X_test, y_test)))

y_dtc = classifier.predict(X_test)

print('accuracy_score of decesion tree classifier: ', accuracy_score(y_dtc, y_test))
print('confusion_matrix of decision tree classifier: ', confusion_matrix(y_dtc, y_test))
print('precision_score of decision tree classifier: ', precision_score(y_dtc, y_test))
print('recall_score of decision tree classifier: ', recall_score(y_dtc, y_test))
print('precision_recall_curve of decision tree classifier: ', precision_recall_curve(y_dtc, y_test))
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print('Accuracy XGBclassifier on train set: {:.3f}'.format(xgb.score(X_train, y_train)))
print('Accuracy XGBClassifier on test set: {:.3f}'.format(xgb.score(X_test, y_test)))
y_xgbc = xgb.predict(X_test)
# predicting Confusion matrix, accuracy score,precision score, recall score
print('accuracy_score of xgboost: ', accuracy_score(y_test, y_xgbc))
print('confusion_matrix of xgboost: ', confusion_matrix(y_test, y_xgbc))
print('precision_score of xgboost: ', precision_score(y_test, y_xgbc))
print('recall_score of xgboost: ', recall_score(y_test, y_xgbc))
print('precision_recall_curve of xgboost: ', precision_recall_curve(y_test, y_xgbc))
# prediction using Naive Bayes Algorithm 
from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()
nbc.fit(X_train, y_train)

print('accuracy of Naive Bayes training set: {:.3f}'.format(nbc.score(X_train,y_train)))
print('accuaracy of Naive Bayes test set: {:.3f}'.format(nbc.score(X_test, y_test)))
y_nb = nbc.predict(X_test)

print('accuracy_score of Naive Bayes: ', accuracy_score(y_test, y_nb))
print('confusion_matrix of Naive Bayes: ', confusion_matrix(y_test, y_nb))
print('precision_score of Naive Bayes: ', precision_score(y_test, y_nb))
print('recall_score of Naive Bayes: ', recall_score(y_test, y_nb))
print('precision_recall_curve of Naive Bayes: ', precision_recall_curve(y_test, y_nb))
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 15)

classifier.summary()