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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
original = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx',"Data")
original.head()
original.info()
df = original

df.drop(['ID'],axis = 1, inplace = True)

x = df.drop(['Personal Loan'], axis = 1)

y = df['Personal Loan']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
resultados=[]

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

for i in range(10):

    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.1*i+0.1).fit(x_train, y_train)

    y_pred=clf.predict(x_test)

    resultados.append([i*0.1,accuracy_score(y_test, y_pred)])

resultados=pd.DataFrame(resultados)

resultados.columns=['learning_rate','accuracy']

plt.plot(resultados.learning_rate,resultados.accuracy)

plt.xlabel('Learning_rate')

plt.ylabel('Accuracy')

plt.title('Ada Boost Classifier')

plt.show()
optimo=min(resultados[resultados.accuracy==max(resultados.accuracy)].learning_rate)

optimo
clf = AdaBoostClassifier(n_estimators=100,learning_rate=optimo).fit(x_train, y_train)

y_pred=clf.predict(x_test)

accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test,y_pred)

matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))