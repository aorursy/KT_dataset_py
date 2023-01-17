# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/tips.csv', encoding = 'iso8859_2')

# Any results you write to the current directory are saved as output.
cat_var = df.dtypes.loc[df.dtypes=='object'].index

le = LabelEncoder()

for cat in cat_var:

  df[cat] = le.fit_transform(df[cat])  
def rebalance(x):

    if x.UID > 10000 and x.Result == 0:

        return False

    return True



df['TipsterActive'] = df.apply(rebalance, axis=1)



df_train = df[df.TipsterActive == True]

X = df_train[['Tipster', 'Track', 'Horse', 'Bet Type', 'Odds']]

y = df_train.Result.values



print('# of win', np.where(y == 1)[0].size)

print('# of lose', np.where(y == 0)[0].size)

y.mean()

#Only 19.5% of the results are 'Win', which means that you could obtain 80.5% accuracy by always predicting "Lose". 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn import cross_validation



clf = LogisticRegression()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)*100

print('Accuracy of %r Classifier = %2f' % (clf, accuracy) + ' %')

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report



y_pred = clf.predict(X_test)

probs = clf.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))

print(roc_auc_score(y_test, probs[:, 1]))
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
clf.predict_proba(np.array([8, 2, 10554, 1, 1.33]).reshape(-1, 5))