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

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
hours = df['Time']/3600

hours = hours.astype(int)

df['Hours'] = hours
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

df.hist (bins=50, figsize=(40,30), color = 'crimson')



plt.show()




fig, ax = plt.subplots(1 , figsize=(18,4))

sns.distplot(hours, color='dodgerblue', bins = 50)

ax.set_title('Distribution of Transaction Time', fontsize=14)







plt.show()
a= len(df[df['Class'] == 0] )

print ("Amount of Non Fraud transactions = " , a)

b = len(df[df['Class'] == 1])

print ("Amount of Fraud transactions = " ,b )
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('Class', data=df)

plt.title('Equally Distributed Classes', fontsize=14)

plt.ylabel("Frequency")

plt.show()


ratio = [ a, b] 

title = "Not Fraud" , "Fraud"



plt.figure(figsize=(9,9))

plt.pie(ratio, labels= title, shadow=True, startangle=0)

plt.title('Pie Chart Ratio of Transactions by their Class\n', fontsize=16)

sns.set_context("paper", font_scale=1.2)



from sklearn.model_selection import train_test_split

from sklearn.utils import resample



from imblearn.over_sampling import SMOTE



# Separate input features and target

Y = df.Class

X = df.drop(['Time','Class',], axis=1)



# setting up testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2727)



sm = SMOTE(random_state=2727, ratio=1.0)

X_train, Y_train = sm.fit_sample(X_train, Y_train)
X_train = pd.DataFrame(data=X_train)

X_train.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',

       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',

       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',

       'Hours']

Y_train = pd.DataFrame(data = Y_train)

Y_train.columns = ['Class']

sns.countplot('Class', data=Y_train)

plt.title('Equally Distributed Classes', fontsize=14)

plt.ylabel("Frequency")

plt.show()
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(X_train, Y_train)



# Predict on test

smote_pred = smote.predict(X_test)

# predict probabilities

probs = smote.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]
accuracy = accuracy_score(Y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(X_train, Y_train)

prediction=log_model.predict(X_test)

score= accuracy_score(Y_test, prediction)

print("Test Accuracy is" , score*100)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 100)



clf.fit(X_train , Y_train)
pred=clf.predict(X_test)

sc= accuracy_score(Y_test, pred)

print("Test accuracy is " , sc*100)