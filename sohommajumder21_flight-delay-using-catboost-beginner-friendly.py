import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import roc_auc_score
train = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip")

test = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip")
train.head(10)
test.head(10)
# changing target to numerical: N to 0 & Y to 1

train.loc[(train.dep_delayed_15min == 'N'), 'dep_delayed_15min'] = 0

train.loc[(train.dep_delayed_15min == 'Y'), 'dep_delayed_15min'] = 1
# Clean month, day of month and day of week

train['Month'] = train['Month'].str[2:].astype('int')

train['DayofMonth'] = train['DayofMonth'].str[2:].astype('int')

train['DayOfWeek'] = train['DayOfWeek'].str[2:].astype('int')



# Check the results

train.head(15)
# Clean month, day of month and day of week

test['Month'] = test['Month'].str[2:].astype('int')

test['DayofMonth'] = test['DayofMonth'].str[2:].astype('int')

test['DayOfWeek'] = test['DayOfWeek'].str[2:].astype('int')



# Check the results

test.head(15)
from sklearn.preprocessing import LabelEncoder



lb= LabelEncoder()

train["UniqueCarrier_new"] = lb.fit_transform(train["UniqueCarrier"])

train[["UniqueCarrier_new", "UniqueCarrier"]].head(11)
train["Origin_new"] = lb.fit_transform(train["Origin"])

train["Dest_new"] = lb.fit_transform(train["Dest"])
train.head()
X= train[['Month','DayofMonth','DayOfWeek','DepTime','Distance','UniqueCarrier_new','Origin_new','Dest_new']]

y= train['dep_delayed_15min']

y=y.astype('int')
#cleaning the test set

test["Origin_new"] = lb.fit_transform(test["Origin"])

test["Dest_new"] = lb.fit_transform(test["Dest"])

test["UniqueCarrier_new"] = lb.fit_transform(test["UniqueCarrier"])



test = test.drop(['UniqueCarrier','Origin','Dest'],1)
test.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)
from catboost import CatBoostClassifier

classifier = CatBoostClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn import metrics



print(metrics.classification_report(y_test,y_pred))
accuracy = classifier.score(y_pred,y_test)

print(accuracy*100,'%')
test_pred = classifier.predict_proba(X_test)[:,1]
roc_auc_score(y_test,test_pred )

test_pred
predictions = classifier.predict_proba(test)[:, 1]

predictions

submission = pd.DataFrame({'id':range(100000),'dep_delayed_15min':predictions})

submission.head(1001)
filename = 'Flight_delay_predictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)