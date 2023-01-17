import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))
#sessions = pd.read_csv('../input/sessions.csv')
test = pd.read_csv('../input/test_users.csv')
train = pd.read_csv('../input/train_users_2.csv')

train.describe()
df_train = train[['id', 'gender' ,'date_account_created', 'age', 'language']].copy()
df_test = test[['id', 'gender' ,'date_account_created', 'age', 'language']].copy()


all_data = pd.concat((df_train, df_test))

all_data['gender'].replace('-unknown-',np.nan, inplace=True)

all_data.isnull().sum()
all_data.info()


all_data['gender']=all_data['gender'].map({"FEMALE":1, "MALE":0})
avg = all_data['gender'].median()
all_data['gender'] = all_data['gender'].replace(np.nan, avg)
train.info()


ageMedian = all_data['age'].median()
all_data['age'] = all_data['age'].replace(np.nan, ageMedian)


from sklearn.preprocessing import LabelEncoder
languageEncoder = LabelEncoder()
all_data['language'] = languageEncoder.fit_transform(all_data['language'])

'''
Now, it is time to merge two columns and data wrangling part
'''

X_train = all_data[:train.shape[0]]
X_train = X_train.drop(['id', 'date_account_created'], axis = 1)
X_test = all_data[train.shape[0]:]
y = train.country_destination

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}




y = y.map(country_num_dic)
y_meadina = y.median()

y = y.replace(np.nan, y.median())

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor()
reg.fit(X_train, y)#train, y
X_test = X_test.drop(['id', 'date_account_created'], axis=1)
pred = reg.predict(X_test)#test

Y_pred = pred.astype(int)


Y_pred = pd.Series(Y_pred).map(num_country_dic)
submission = pd.DataFrame({'id': test.id, 'country':Y_pred})
submission.to_csv('submission.csv', index=False)