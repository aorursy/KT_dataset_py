import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/von-willebrand/VWD.csv")
df.head()
df.shape
df.dtypes
#ดูข้อมูลที่เป็นตัวเลข

df.describe(include=[np.number]) 
#ดูข้อมูลที่ไม่ใช่ตัวเลข

df.describe(include = [np.object])
#Checking for null values in the data.



df.isnull().sum()
df['target'].value_counts()
plt.bar(df['target'].value_counts(dropna=False).index.tolist(),

  df['target'].value_counts(dropna=False).values.tolist(),

  color=['green', 'yellow','red'])

plt.title("von Willebrand Disease ")

plt.ylabel('Frequency')

plt.xlabel('Types of von Willebrand Disease ')

plt.tight_layout()
sns.set_style('whitegrid')



plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='aPTT ratio',y='Factor VIII',data=df,hue='target')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='VWF_Antigen',y='VWF_CBA',data=df,hue='target')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='VWF_Antigen',y='VWF_GPIbM',data=df,hue='target')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='VWF_CBA',y='VWF_GPIbM',data=df,hue='target')

plt.show()
df.describe(include='all')
#Display columns that contain null values.

df.isnull().any()
df.info()
cols = df.select_dtypes(include=['float64', 'int64']).columns[ :-1]

print(cols)
numeric_cols = ['age', 'aPTT ratio','Factor VIII', 'VWF_GPIbM', 'VWF_Antigen']
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer



scaler = StandardScaler()

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])



df.head()
X = df.iloc[:,0:-1]

y = df.iloc[:,-1]
print(X.shape)
print(y.shape)
y.value_counts()
#Display only categorical columns

  

categorical_columns = X.select_dtypes(include=['object']).columns

print(categorical_columns)
for column in categorical_columns:

  dummies = pd.get_dummies(X[column], prefix=column, drop_first=False)

  X = pd.concat([X, dummies], axis=1)
X.shape
X.head()
#Remove original categorical columns



X.drop(categorical_columns, inplace=True, axis=1)
X.shape
X.head()
#Label encoding on the 'target' column



from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

le = LabelEncoder()

y = le.fit_transform(y)
#Convert an array to a Series



y = pd.Series(y)

y.value_counts()
#Create a data frame from a Series



y = pd.DataFrame(y, columns=['target'])

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2019)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif
#Select features 

selector = SelectKBest(score_func=mutual_info_classif, k=13)



#Fit the selector to the training data set.

selector_model = selector.fit(X_train, y_train)
print(selector_model.scores_[selector_model.get_support(indices=True)])
from sklearn.neural_network import MLPClassifier



mymodel_NeuralNetwork = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu',solver='adam', max_iter=500)

mymodel_NeuralNetwork.fit(X_train, y_train)



y_pred = mymodel_NeuralNetwork.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')





print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
from sklearn import svm



mymodel_svm = svm.SVC(kernel='linear')

mymodel_svm.fit(X_train, y_train)



y_pred = mymodel_svm.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
from sklearn.neighbors import KNeighborsClassifier



mymodel_KNeighbors = KNeighborsClassifier(n_neighbors=3)

mymodel_KNeighbors.fit(X_train, y_train)



y_pred = mymodel_KNeighbors.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))