import pandas as pd
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train = df_train.drop(['Dates', 'Descript', 'Resolution', 'Address'], axis=1)

df_test = df_test.drop(['Id','Dates', 'Address'], axis=1)

df_train.head()
## using label encoder
from sklearn.preprocessing import LabelEncoder



le_cat = LabelEncoder()

cat_encoded = le_cat.fit_transform(df_train['Category'])

#cat_encoded_t = le_cat.fit_transform(df_test['Category'])



le_day = LabelEncoder()

day_encoded = le_day.fit_transform(df_train['DayOfWeek'])

day_encoded_t = le_day.fit_transform(df_test['DayOfWeek'])



le_district = LabelEncoder()

district_encoded = le_district.fit_transform(df_train['PdDistrict'])

district_encoded_t = le_district.fit_transform(df_test['PdDistrict'])



## apply label encoded on dataset



df_train['Category'] = cat_encoded

df_train['DayOfWeek'] = day_encoded

df_train['PdDistrict'] = district_encoded



df_test['DayOfWeek'] = day_encoded_t

df_test['PdDistrict'] = district_encoded_t



df_train.head()

#df_test.head()
from sklearn.model_selection import train_test_split



y = df_train['Category']

X = df_train.drop(['Category'], axis=1)



# split data

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
test_X = df_test
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.metrics import log_loss

#print(X_test)

    

dt = DecisionTreeClassifier(max_depth=10, random_state=42)

dt.fit(X_train,y_train)

#y_pred = dt.predict(X_test)

y_pred = dt.predict_proba(X_test)

print(log_loss(y_test, y_pred))
dt.feature_importances_
# creating a sample dataset

y_pred = dt.predict(test_X)

le_cat.inverse_transform(y_pred)
# probability

import numpy as np

y_pred_proba = dt.predict_proba(test_X)



probs = np.zeros((1,39))

probs = y_pred_proba

#print(probs)

col = le_cat.inverse_transform(range(39))

print(col)
# criando um dataframe



ds = pd.DataFrame(probs, columns=col)

counts = np.array(range(ds.shape[0]))

counts = counts.astype('int64')





ds.insert(0, "Id", counts)

ds.info()
# save dataframe submission

ds.to_csv('submission.csv', sep=',', index=False)