import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
credit = pd.read_csv(r'creditcard.csv')
credit.head()
print('Colunas', credit.shape[1])

print('Linhas', credit.shape[0])

print('Dados Unicos', credit.nunique())
print('Numero de Valores Nulos',credit.isnull().sum().sum())
sns.set_style('whitegrid')

plt.figure(figsize = (10,7))



sns.countplot('Class',data = credit,palette = 'Set3')
print('Transações Não Fraudulentas', round(credit['Class'].value_counts()[0] / len(credit['Class']),3) * 100,'%')

print('Transações Fraudulentas', round(credit['Class'].value_counts()[1] / len(credit['Class']),3) * 100,'%')
credit.info()
fraud = credit.loc[credit['Class'] == 1,'Amount']

no_fraud = credit.loc[credit['Class'] == 0,'Amount']
plt.figure(figsize = (8,4))

sns.distplot(fraud,color = 'r')

plt.xlabel('Amount Fraudulento',size = 15)
plt.figure(figsize = (8,4))

sns.distplot(no_fraud,color = 'g')

plt.xlabel('Amount Não Fraudulento',size = 15)
time_fraud = credit.loc[credit['Class'] == 1 ,'Time']

time_nofraud = credit.loc[credit['Class'] == 0 ,'Time']
plt.figure(figsize = (8,4))

sns.distplot(time_fraud,color = 'r')

plt.xlabel('Tempo_Transação Fraudulenta',size = 15)
plt.figure(figsize = (8,4))

sns.distplot(time_nofraud,color = 'g')

plt.xlabel('Tempo_Transação Não Fraudulenta',size = 15)
from sklearn.preprocessing import StandardScaler, Normalizer
scaler = StandardScaler()



credit['Time2'] = scaler.fit_transform(credit['Time'].values.reshape(-1,1))

credit['Amount2'] = scaler.fit_transform(credit['Amount'].values.reshape(-1,1))
credit.drop(['Time','Amount'],axis = 1,inplace = True)
n_credit = credit.copy()
from imblearn.over_sampling import SMOTE

sm = SMOTE()
n_credit['Class'].value_counts()
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score



x = n_credit.drop('Class',axis = 1)

y = n_credit['Class']



x,y = sm.fit_sample(x,y)

    

np.bincount(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 101,test_size = 0.2)
lm = LogisticRegression()



lm.fit(x_train,y_train)

lm.score(x_train,y_train)



y_pred = lm.predict(x_test)

print(accuracy_score(y_test, y_pred))
svm = SVC()

svm.fit(x_train,y_train)

svm.score(x_train,y_train)
print(accuracy_score(y_test,y_pred))