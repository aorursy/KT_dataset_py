# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the 
import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter("ignore")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
bank_df=pd.read_excel("/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx",'Data')

bank_df.head()
bank_df.shape
bank_df.info()
bank_df.describe()
bank_df.columns
bank_df.isnull().sum()
categorical=[col for col in bank_df.columns if bank_df[col].nunique()<=5]

continous=[col for col in bank_df.columns if bank_df[col].nunique()>5]

print(categorical)

print(continous)
categorical.remove("Personal Loan")

continous.remove("ID")

print(categorical)

print(continous)
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(continous):

    ax=fig.add_subplot(2,3,i+1)

    sns.distplot(bank_df[col])
fig=plt.figure(figsize=(20,10))

for i,col in enumerate(categorical):

    ax=fig.add_subplot(2,3,i+1)

    sns.countplot(bank_df[col])
sns.pairplot(bank_df,

    x_vars=['Age', 'Experience', 'Income', 'ZIP Code', 'CCAvg', 'Mortgage'],

    y_vars=['Age', 'Experience', 'Income', 'ZIP Code', 'CCAvg', 'Mortgage'],

             diag_kind="kde",hue="Personal Loan")

plt.show()
bank_df.drop_duplicates(inplace=True)
bank_df.shape
bank_df.set_index("ID",inplace=True)
bank_df.head()
bank_df.drop("ZIP Code",axis=1,inplace=True)
corr=bank_df.corr()

plt.figure(figsize=(10,10))

plt.title('Correlation')

ax=sns.heatmap(corr, annot=True, square=True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
bank_df.drop('Experience',axis=1,inplace=True)
bank_df['Extra_serv']=bank_df['Online']+bank_df['CreditCard']
bank_df[['Extra_serv','Online','CreditCard','Personal Loan']].corr()
bank_df.drop(['Online','CreditCard'],axis=1,inplace=True)
bank_df.head()
X = bank_df.drop('Personal Loan', axis=1)

y = bank_df['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train.head()
scaler=StandardScaler()
X_train[['Age','Income','Family','CCAvg','Education','Mortgage','Extra_serv']]=scaler.fit_transform(X_train[['Age','Income','Family','CCAvg','Education','Mortgage','Extra_serv']])

X_train.head()
X_test[['Age','Income','Family','CCAvg','Education','Mortgage','Extra_serv']]=scaler.fit_transform(X_test[['Age','Income','Family','CCAvg','Education','Mortgage','Extra_serv']])

X_test.head()
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)
tree_score=f1_score(y_test,y_test_pred)

tree_score
print(accuracy_score(y_train, y_train_pred))

confusion_matrix(y_train, y_train_pred)
print(accuracy_score(y_test, y_test_pred))

confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test,y_test_pred))