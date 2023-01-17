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
!wget 'https://raw.githubusercontent.com/amankharwal/Website-data/master/payment_fraud.csv'
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv('payment_fraud.csv')
df.head()
df.columns
df.paymentMethod.unique()
df.paymentMethod.replace('paypal',0,inplace=True)

df.paymentMethod.replace('storecredit',1,inplace=True)

df.paymentMethod.replace('creditcard',1,inplace=True)



# Split dataset up into train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    df.drop('label', axis=1), df['label'],

    test_size=0.33, random_state=17)
clf = LogisticRegression().fit(X_train, y_train)



# Make predictions on test set

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred, y_test))
# Compare test set predictions with ground truth labels

print(confusion_matrix(y_test, y_pred))