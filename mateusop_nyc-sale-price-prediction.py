import pandas as pd

import numpy as np

import seaborn as sns

import math

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_valid = pd.read_csv('../input/valid.csv')

df_test = pd.read_csv('../input/test.csv')
df_valid_test = pd.concat([df_valid,df_test])
submission = pd.DataFrame()

submission['sale_id'] = df_valid_test['sale_id']
df_raw = pd.concat([df_train,df_valid_test],sort=True)

print(len(df_valid_test))

print(len(df_train))

print(len(df_raw))
def display_all(df):

    with pd.option_context("display.max_rows", 1000):

        with pd.option_context("display.max_columns", 1000):

            display(df)
display_all(df_raw.transpose())
df_raw = df_raw.drop(['sale_id', 'ease-ment', 'apartment_number'], axis=1)
df_raw.shape
df_raw.dtypes
df_raw = df_raw.replace(to_replace='-', value=0, regex=True)



df_raw['neighborhood'].fillna(df_raw['neighborhood'].mode()[0], inplace=True)

df_raw['building_class_category'].fillna(df_raw['building_class_category'].mode()[0], inplace=True)

df_raw['address'].fillna(df_raw['address'].mode()[0], inplace=True)

df_raw['land_square_feet'].fillna(df_raw['land_square_feet'].mode()[0], inplace=True)

df_raw['gross_square_feet'].fillna(df_raw['gross_square_feet'].mode()[0], inplace=True)

df_raw['building_class_category'].fillna(df_raw['building_class_category'].mode()[0], inplace=True)



df_raw['sale_date'] = pd.to_datetime(df_raw.sale_date, format='%m/%d/%y').astype(int)



df_raw['land_square_feet'] = pd.to_numeric(df_raw['land_square_feet'], errors='coerce')

df_raw['gross_square_feet']= pd.to_numeric(df_raw['gross_square_feet'], errors='coerce')



df_raw['land_square_feet'] = df_raw['land_square_feet'].replace(to_replace=0, value=df_raw['land_square_feet'].mean(), regex=True)

df_raw['gross_square_feet'] = df_raw['gross_square_feet'].replace(to_replace=0, value=df_raw['gross_square_feet'].mean(), regex=True)
corr = df_raw.corr()

sns.heatmap(corr)

corr['sale_price'].sort_values(ascending=False)
cat_data = df_raw.select_dtypes(exclude=[np.number])

cat_data.describe()
pivot = df_raw.pivot_table(index='tax_class_at_time_of_sale', values='sale_price', aggfunc=np.median)

pivot.plot(kind='bar')

pivot
pivot = df_raw.pivot_table(index='borough', values='sale_price', aggfunc=np.median)

pivot.plot(kind='bar')

pivot
df_raw.dtypes
df_final = pd.concat([df_raw, pd.get_dummies(df_raw['building_class_at_present'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['tax_class_at_time_of_sale'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['tax_class_at_present'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['borough'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['neighborhood'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['building_class_category'])], axis=1);

df_final = pd.concat([df_raw, pd.get_dummies(df_raw['building_class_at_time_of_sale'])], axis=1);

df_final = df_final.drop(['address', 'borough', 'neighborhood','building_class_category','tax_class_at_present',

                          'building_class_at_present', 'tax_class_at_time_of_sale', 'building_class_at_time_of_sale', 'sale_date'], axis=1)
df_submission = df_final[df_final.isnull().any(axis=1)]

df_train = df_final[df_final['sale_price'] >= 0]



X = df_train.drop(['sale_price'], axis=1)

y = df_train['sale_price']



x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = RandomForestRegressor(n_estimators=30, n_jobs=-1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('R2 {}'.format(model.score(x_test, y_test)))
df_submission = df_submission.drop('sale_price', axis = 1)
final_model = RandomForestRegressor(n_estimators=30, n_jobs=-1)

final_model.fit(X, y)

submission['sale_price'] = final_model.predict(df_submission)
submission.to_csv('submission.csv', index = False)