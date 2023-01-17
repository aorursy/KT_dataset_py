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
loan_data = pd.read_csv('/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')

loan_data.head()
loan_data.info()
#Dropping the 'ID' column, as it is not used for the prediction

loan_data.drop('ID',axis = 1, inplace = True)

loan_data.head()
#defining features and label

x = loan_data.drop(['Personal Loan'],axis = 1)

y = loan_data['Personal Loan']
x.info()
#splitting train and test set

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.naive_bayes import GaussianNB

model_gnb = GaussianNB()

model_gnb.fit(x_train,y_train)

y_pred_gnb = model_gnb.predict(x_test)

y_pred_gnb

print("Train_Score: ", model_gnb.score(x_train, y_train)*100)

print("Test_Score: ", model_gnb.score(x_test, y_test)*100)
from sklearn.linear_model import LogisticRegression



model_lr = LogisticRegression(random_state=1)

model_lr.fit(x_train,y_train)



y_pred_lr = model_lr.predict(x_test)



print("Train_Score: ", model_lr.score(x_train, y_train)*100)

print("Test_Score: ", model_lr.score(x_test, y_test)*100)