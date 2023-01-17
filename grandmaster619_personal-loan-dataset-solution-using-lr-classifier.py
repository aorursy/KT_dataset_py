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
data_master = pd.read_csv('/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')
data_master.info()
data_master.rename(columns = {'ZIP Code' : 'ZIP_Code', 'Personal Loan' : 'Personal_Loan', 'Securities Account' : 'Securities_Account',  'CD Account' : 'CD_Account',}, inplace = True)
X = data_master.drop(['Personal_Loan'], axis = 1)

y = data_master.Personal_Loan


from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(logreg.score(X_train, y_train)))

print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))

y_pred=logreg.predict(X_test)

print("Classification result is :{}".format(y_pred[:]))

from sklearn.metrics import recall_score, precision_score, confusion_matrix
print("Recall score", recall_score(y_test, y_pred, average='macro'))

print("Precision score", precision_score(y_test, y_pred, average='macro'))

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred))