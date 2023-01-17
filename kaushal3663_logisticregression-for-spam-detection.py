# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math
emails = pd.read_csv('/kaggle/input/E-mails.csv')
emails.head()
emails.shape
print("no of emails"+str(len(emails.index)))
sns.countplot(x = 'is_spam',data=emails)
sns.countplot(x = 'bang',data=emails)
sns.countplot(x = 'dollar',data=emails)
sns.countplot(x = 'money',data=emails)
sns.countplot(x = 'crl.tot',data=emails)
sns.countplot(x = 'n000',data=emails)
sns.countplot(x = 'make',data=emails)
emails.info
emails.isnull()
emails.head(3)
x=emails.drop('is_spam',axis =1)

y= emails['is_spam']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state =1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 
logmodel.fit(x_train,y_train)
prediction = logmodel.predict(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,prediction)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)