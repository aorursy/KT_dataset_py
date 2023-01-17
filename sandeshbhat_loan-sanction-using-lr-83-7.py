# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

dataframe = dataframe.drop(columns=["Loan_ID"])

dataframe.head()
len(dataframe)
dataframe.info()
dataframe.isnull().sum()
from sklearn.preprocessing import LabelEncoder

cat_features=[x for x in dataframe.columns if dataframe[x].dtype=="object"]

le=LabelEncoder()

for col in cat_features:

    if col in dataframe.columns:

        i = dataframe.columns.get_loc(col)

        dataframe.iloc[:,i] = dataframe.apply(lambda i:le.fit_transform(i.astype(str)), axis=0, result_type='expand')

dataframe.head(10)
X = dataframe.iloc[:, :-1].values

y = dataframe.iloc[:, -1].values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer.fit(X[:, 7:8])

X[:, 7:8] = imputer.transform(X[:, 7:8])

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(X)

X = imputer.transform(X)
plt.figure(figsize=(10,10))

print(sns.heatmap(dataframe.corr(), annot=True, fmt='.2f'))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, 5:9] = sc.fit_transform(X_train[:, 5:9])

X_test[:, 5:9] = sc.transform(X_test[:, 5:9])
print(X_train)
print(X_test)
def logistic_regression(X_train,X_test,y_train,y_test):

  from sklearn.linear_model import LogisticRegression

  from sklearn.metrics import accuracy_score,classification_report

  LR = LogisticRegression(random_state = 0)

  LR.fit(X_train,y_train)

  print("Logistic Regression\n",classification_report(y_test, LR.predict(X_test)),"\n")

  print(accuracy_score(y_test, LR.predict(X_test)),"\n")



def SVM(X_train,X_test,y_train,y_test):

  from sklearn.svm import SVC 

  from sklearn.metrics import accuracy_score,classification_report

  svc = SVC(kernel = 'linear', random_state = 0)

  svc.fit(X_train,y_train)

  print("SVM\n",classification_report(y_test, svc.predict(X_test)),"\n")

  print(accuracy_score(y_test, svc.predict(X_test)),"\n")





def kernSVM(X_train,X_test,y_train,y_test):

  from sklearn.svm import SVC 

  from sklearn.ensemble import AdaBoostClassifier

  from sklearn.metrics import accuracy_score,classification_report

  kernelsvm = SVC(kernel = 'rbf', random_state = 42)

  kernelsvm.fit(X_train,y_train)

  print("KernSVM\n",classification_report(y_test, kernelsvm.predict(X_test)),"\n")

  print(accuracy_score(y_test, kernelsvm.predict(X_test)),"\n")



def Naive_Bayes(X_train,X_test,y_train,y_test):

  from sklearn.naive_bayes import GaussianNB 

  from sklearn.metrics import accuracy_score,classification_report

  NB = GaussianNB()

  NB.fit(X_train,y_train)

  print("Naive_Bayes\n",classification_report(y_test, NB.predict(X_test)),"\n")

  print(accuracy_score(y_test, NB.predict(X_test)),"\n")



logistic_regression(X_train,X_test,y_train,y_test)

SVM(X_train,X_test,y_train,y_test)

kernSVM(X_train,X_test,y_train,y_test)

Naive_Bayes(X_train,X_test,y_train,y_test)