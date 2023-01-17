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

df_train = pd.read_csv('/kaggle/input/creditscoringds/df_train.csv', sep=',')

df_test = pd.read_csv('/kaggle/input/creditscoringds/df_test.csv', sep=',')

#df_test = df_test.drop(['ProductCategory_ticket'], axis=1)



tr_id = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

tr_id.head()
df_train.head()
y_test = tr_id[['TransactionId']]

y_test['IsDefaulted'] = 1

y_test.head()
X_test = df_test

X_test.head()
df_train.head()
'''from sklearn.utils import shuffle

df_train = shuffle(df_train)

df_train.head()'''
from sklearn.model_selection import train_test_split

X_all = df_train.drop('IsDefaulted', axis=1)

y_all = df_train['IsDefaulted']

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 

                                                     test_size=0.3, 

                                                     random_state=17)
from sklearn.linear_model import LogisticRegressionCV

log_c = LogisticRegressionCV(cv=5, random_state=17)

log_c.fit(X_all, y_all)



from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(random_state=17,

                              penalty='elasticnet',

                              solver='saga',

                              l1_ratio=0.5,

                              class_weight={0:0.4, 1:0.6})



#X = (np.asarray(df.X)).reshape(-1, 1)

#Y = (np.asarray(df.Y)).ravel()



logistic.fit(X_train, y_train)

print(logistic.score(X_valid, y_valid))

y_pred = logistic.predict(X_valid)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)
from sklearn.metrics import accuracy_score

print('Accuracy:', accuracy_score(y_valid, y_pred))



from sklearn.metrics import f1_score

print('F-score:', f1_score(y_valid, y_pred))
logistic.fit(X_all, y_all)
#y_test['IfDefaulted'] = logistic.predict(df_test)

#y_test.head()

y_test_c = y_test

y_x_c = log_c.predict(df_test)

for i in range(y_test_c.shape[0]):

    y_test_c.at[i,'IsDefaulted'] = y_x_c[i]

y_test_c.head()





y_x = logistic.predict(df_test)

for i in range(y_test.shape[0]):

    y_test.at[i,'IsDefaulted'] = y_x[i]

y_test.head()
y_test.to_csv('ANSWER_log.csv',index=False)

y_test_c.to_csv('ANSWER_log_c).csv',index=False)
import pandas as pd

df_test_final = pd.read_csv("../input/cs-ds/df_test_final.csv")

df_train_final = pd.read_csv("../input/cs-ds/df_train_final.csv")