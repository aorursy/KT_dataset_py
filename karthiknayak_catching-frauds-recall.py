import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



data = pd.read_csv('../input/creditcard.csv')
# Lets get a basic idea of the data.



data.describe()
data['Class'].value_counts()
# We can see that the no of frauds is quite low. 



print('fraud_percentage:', (492/(492 + 284315)))
# Guessing 0 would give us an accuracy of



print('guessing 0 accuracy:', (284315/(284315 + 492)))
# Lets look at the correlation of the data with the Class feature



sr = data.corr()['Class']

sr.drop(['Class'], inplace=True)

df = pd.DataFrame({'feature': sr.index, 'correlation': sr.values})

sns.barplot(y='correlation', x='feature', data=df)

plt.xticks(rotation=90)

plt.show()
# Lets sort this and make all values positive



df['correlation'] = df['correlation'].apply(lambda x: abs(x))

df.sort_values(['correlation'], axis=0, ascending=False, inplace=True)

sns.barplot(y='correlation', x='feature', data=df)

plt.xticks(rotation=90)

plt.show()
# Drop the last few features with really low correlation



features_to_drop = df[-5:]['feature'].values

data.drop(features_to_drop, axis=1, inplace=True)
# Let's split the data now



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data.drop(['Class'], axis=1), data['Class'], test_size=0.3,

                                                    random_state=32, stratify=data['Class'])
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



clf = AdaBoostClassifier(n_estimators=400, learning_rate=1.1)

clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))



array = confusion_matrix(y_test, clf.predict(X_test))

df_cm = pd.DataFrame(array, index=['Actually Negative', 'Actually Positive'], columns=['Predicted Negative', 'Predicted Postive'])

sns.heatmap(df_cm, annot=True, fmt='g')

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))



array = confusion_matrix(y_test, clf.predict(X_test))

df_cm = pd.DataFrame(array, index=['Actually Negative', 'Actually Positive'], columns=['Predicted Negative', 'Predicted Postive'])

sns.heatmap(df_cm, annot=True, fmt='g')

plt.show()