import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel ("../input/credit-risk-predicting-loan-default/dataDT.xls")
df.info()
df.head()
df['default'].unique()
df['branch'].unique()
df.groupby(['branch']).count()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.hist(df['age'])
plt.figure(figsize=(11,7))

sns.lmplot(y='creddebt',x='age',data=df,hue='default')
plt.figure(figsize=(11,7))

sns.lmplot(y='income',x='age',data=df,hue='default')
plt.figure(figsize=(11,7))

sns.lmplot(y='creddebt',x='employ',data=df,hue='default')
X=df[['age','employ','debtinc','creddebt']]

y=df['default']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
print(logmodel.intercept_)
print(logmodel.coef_)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)