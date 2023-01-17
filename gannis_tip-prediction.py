import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import datetime, time
df = pd.read_csv('../input/1january.csv')
df.head(20)
df.info()
sns.jointplot(x='fare_amount', y='tip_amount', data=df)
df = df.drop('pickup_latitude',1)
df = df.drop('pickup_longitude',1)
df = df.drop('dropoff_latitude',1)
df = df.drop('dropoff_longitude',1)
df = df.drop(df[df['fare_amount'] > 1000].index)   # Abnormally expensive fairs. Don't think anyone is really spending $100,000 on a taxi ride
df = df.drop(df[df['fare_amount'] < 0].index)    # Negative fairs. What, taxi driver got mugged? Don't think so.
df = df.drop(df[df['tip_amount'] < 0].index)    # Negative tips. Taxi driver mugged passenger?
df = df.drop(df[df['payment_amount'] < 0].index)   # Somehow I don't see cabbies giving money away.
df = df.drop(df[df['passenger_count'] <= 0].index)    # Drop all entries with 0 passengers.
def convert_day(x):
    d = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()
    return d.day % 7
df['day'] = df['pickup_datetime'].apply(lambda x: convert_day(x))
def convert_time(x):
    t = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time()
    hour = t.hour
    if hour >= 18 and hour <= 23:
        return 0
    elif hour >= 0 and hour <= 5:
        return 1
    elif hour >= 6 and hour <= 11:
        return 2
    elif hour >= 12 and hour <= 17:
        return 3
df['shift'] = df['pickup_datetime'].apply(lambda x: convert_time(x))
df.tail()
df = df.drop('pickup_datetime',1)
df = df.drop('dropoff_datetime',1)
sns.jointplot(x='fare_amount', y='tip_amount', data=df)
total_tips = df['tip_amount'].sum()
average_tip = total_tips / len(df.index)
print('Average tip: ' + str(average_tip))
did_tip = []
for val in df['tip_amount']:
    if val > 0:
        did_tip.append(1)
    else:
        did_tip.append(0)
did_tip = pd.Series(did_tip)
df['did_tip'] = did_tip.values
total_tips = 0
for val in df['did_tip']:
    if val:
        total_tips += 1
        
print(total_tips)
print('chance of tip: ' + str(total_tips/len(df['did_tip'])))
from sklearn.cross_validation import train_test_split
df_feat = df.drop('did_tip', 1)
df_target = df['did_tip']
from sklearn import linear_model
from sklearn.metrics import classification_report,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(df_feat, df_target, test_size=0.30)
model = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight='balanced', epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='hinge', n_iter=10, n_jobs=3,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(predictions)
print(classification_report(y_test,predictions))
april = pd.read_csv('../input/4april.csv')
df = None
april = april.drop('pickup_latitude',1)
april = april.drop('pickup_longitude',1)
april = april.drop('dropoff_latitude',1)
april = april.drop('dropoff_longitude',1)
april = april.drop(april[april['fare_amount'] > 1000].index)   # Abnormally expensive fairs. Don't think anyone is really spending $100,000 on a taxi ride
april = april.drop(april[april['fare_amount'] < 0].index)    # Negative fairs. What, taxi driver got mugged? Don't think so.
april = april.drop(april[april['tip_amount'] < 0].index)    # Negative tips. Taxi driver tipped passenger?
april = april.drop(april[april['payment_amount'] < 0].index)   # Somehow I don't see cabbies giving money away.
april = april.drop(april[april['passenger_count'] <= 0].index)    # Drop all entries with 0 passengers.
april['day'] = april['pickup_datetime'].apply(lambda x: convert_day(x))
april['shift'] = april['pickup_datetime'].apply(lambda x: convert_time(x))
april = april.drop('pickup_datetime',1)
april = april.drop('dropoff_datetime',1)
did_tip = []
for val in april['tip_amount']:
    if val > 0:
        did_tip.append(1)
    else:
        did_tip.append(0)

did_tip = pd.Series(did_tip)
april['did_tip'] = did_tip.values
april_feat = april.drop('did_tip', 1)
april_target = april['did_tip']
predictions = model.predict(april_feat)
print(confusion_matrix(april_target,predictions))
print(predictions)
print(classification_report(april_target,predictions))