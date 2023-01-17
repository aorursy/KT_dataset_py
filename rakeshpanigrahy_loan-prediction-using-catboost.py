import pandas as pd
import numpy as np
dataset = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
dataset.shape
dataset.count()
dataset.isna().sum()
dataset.dropna(inplace=True)
dataset.isna().sum()
dataset.describe()
dataset['Dependents'].value_counts()
dataset['Dependents'] = dataset['Dependents'].map({'3+': 3, '1':1, '2':2, '0':0})
dataset['Dependents'].value_counts()
dataset.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Married'] = le.fit_transform(dataset['Married'])
dataset['Education'] = le.fit_transform(dataset['Education'])
dataset['Self_Employed'] = le.fit_transform(dataset['Self_Employed'])
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])
dataset.head()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:9] = sc.fit_transform(X_train[:, 5:9])
X_test[:, 5:9] = sc.transform(X_test[:, 5:9])
X_train
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
X_train
!pip install catboost
from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))