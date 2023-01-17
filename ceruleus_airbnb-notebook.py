import pandas as pd

train = pd.read_csv("../input/train_users.csv")
test = pd.read_csv("../input/test_users.csv")
sessions = pd.read_csv("../input/sessions.csv")
for data in (train, test):
    data['year_created']  = data['date_account_created'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[:4]) )
    data['month_created'] = data['date_account_created'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[5:7]))
    data['week_created']  = data['date_account_created'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[8:10]))
    
    data['year_first']  = data['date_first_booking'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[:4]))
    data['month_first'] = data['date_first_booking'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[5:7]))
    data['week_first']  = data['date_first_booking'].apply(lambda x: 'nan' if str(x) == 'nan' else int(str(x)[8:10]))

for data in (train, test):
    data.drop(['date_account_created'], axis=1,inplace=True)
    data.drop(['date_first_booking'], axis=1,inplace=True)

train.head()
train = pd.merge(train, sessions, how="left", left_on=["id"], right_on=["user_id"])
test = pd.merge(test, sessions, how="left", left_on=["id"], right_on=["user_id"])
countries = pd.read_csv("../input/countries.csv")
countries.head()
train = pd.merge(train, countries, how="left", left_on=["country_destination"], right_on=["country_destination"])
x_train = train.drop("country_destination", axis=1)
y_train = train["country_destination"]
x_test = test.copy()
import numpy as np

from sklearn import preprocessing

for f in x_train.columns:
    if x_train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        if f not in x_test.columns:
            lbl.fit(np.unique(list(x_train[f].values)))
            x_train[f] = lbl.transform(list(x_train[f].values))
        else:
            lbl.fit(np.unique(list(x_train[f].values) + list(x_test[f].values)))
            x_train[f] = lbl.transform(list(x_train[f].values))
            x_test[f] = lbl.transform(list(x_test[f].values))
for col in countries.columns:
    if col == 'country_destination':
        continue
    del(x_train[col])
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_train_nonan = imp.fit_transform(x_train)


x_test_nonan = imp.fit_transform(x_test)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


model.fit(x_train_nonan, y_train)
y_test = model.predict(x_test_nonan)
submission = pd.DataFrame()
submission["id"]          = test["id"]
submission["country"] = y_test

submission.to_csv('airbnb.csv', index=False)
