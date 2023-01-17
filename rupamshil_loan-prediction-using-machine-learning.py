import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

dataset.head()   #prints a nutshell of the dataset
dataset.info()  #we get detailed info of the dataset
dataset.shape  #no of rows and columns
dataset.describe()  #prints the numerical columns details
dataset.isnull().sum()
dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].median())
dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset.isnull().sum()
dataset.shape
print(pd.crosstab(dataset['Gender'],dataset['Loan_Status']))
sns.countplot(dataset['Gender'],hue=dataset['Loan_Status'])
print(pd.crosstab(dataset['Married'],dataset['Loan_Status']))

sns.countplot(dataset['Married'],hue=dataset['Loan_Status'])
print(pd.crosstab(dataset['Self_Employed'],dataset['Loan_Status']))

sns.countplot(dataset['Self_Employed'],hue=dataset['Loan_Status'])
print(pd.crosstab(dataset['Property_Area'],dataset['Loan_Status']))

sns.countplot(dataset['Property_Area'],hue=dataset['Loan_Status'])
dataset['Loan_Status'].replace('Y',1,inplace = True)

dataset['Loan_Status'].replace('N',0,inplace = True)
dataset['Loan_Status'].value_counts()
dataset.Gender=dataset.Gender.map({'Male':1,'Female':0})

dataset['Gender'].value_counts()
dataset.Married=dataset.Married.map({'Yes':1,'No':0})

dataset['Married'].value_counts()
dataset.Dependents=dataset.Dependents.map({'0':0,'1':1,'2':2,'3+':3})

dataset['Dependents'].value_counts()
dataset.Education=dataset.Education.map({'Graduate':1,'Not Graduate':0})

dataset['Education'].value_counts()
dataset.Self_Employed=dataset.Self_Employed.map({'Yes':1,'No':0})

dataset['Self_Employed'].value_counts()
dataset.Property_Area=dataset.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})

dataset['Property_Area'].value_counts()
dataset['LoanAmount'].value_counts()
dataset['Loan_Amount_Term'].value_counts()
dataset['Credit_History'].value_counts()
plt.figure(figsize=(16,5))

sns.heatmap(dataset.corr(),annot=True)

plt.title('Correlation Matrix (for Loan Status)')
dataset.head()
X = dataset.iloc[:,1:-1].values

y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print(X_train)
import tensorflow as tf

tf.__version__
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size =32, epochs =100)
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)