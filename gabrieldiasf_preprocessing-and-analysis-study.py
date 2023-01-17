# Setup



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Loading data



brhousedata = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
brhousedata.head()
brhousedata.describe()
brhousedata.isnull().values.any()
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)

sns.distplot(brhousedata.rooms, kde=False)

plt.subplot(3,1,2)

sns.distplot(a=brhousedata.bathroom, kde=False)

plt.subplot(3,1,3)

sns.distplot(a=brhousedata['parking spaces'], kde=False)
plt.figure(figsize=(6,5))

sns.boxplot(x=brhousedata.area)
brhousedata.loc[brhousedata.area>10000]
plt.figure(figsize=(6,5))

sns.boxplot(x=brhousedata['total (R$)'])
brhousedata.loc[brhousedata['total (R$)']>200000]
plt.figure(figsize=(12,12))

sns.scatterplot(x=brhousedata['area'], y=brhousedata['total (R$)'])
brhousedata.loc[(brhousedata['area']>10000) | (brhousedata['total (R$)']>200000)]
brhousedata.drop(brhousedata.loc[(brhousedata['area']>10000) | (brhousedata['total (R$)']>80000)].index, axis=0, inplace=True)
plt.figure(figsize=(12,12))

sns.set(style='white')

sns.scatterplot(x=brhousedata['area'], y=brhousedata['total (R$)'])

sns.regplot(x=brhousedata['area'], y=brhousedata['total (R$)'])
plt.figure(figsize=(8,8))

sns.violinplot(y = brhousedata['total (R$)'], x=brhousedata.furniture, data=brhousedata)

plt.figure(figsize=(8,8))

sns.violinplot(y = brhousedata['total (R$)'], x=brhousedata.animal, data=brhousedata)
plt.figure(figsize=(8,8))

sns.violinplot(y = brhousedata['total (R$)'], x=brhousedata.city, data=brhousedata)
plt.figure(figsize=(8,8))

sns.violinplot(y = brhousedata['area'], x=brhousedata.city, data=brhousedata)
corr_matrix = brhousedata.corr()

corr_matrix
plt.figure(figsize=(7,7))

sns.heatmap(corr_matrix, annot=True)
from sklearn.preprocessing import OneHotEncoder
s = (brhousedata.dtypes == 'object')

object_cols = list(s[s].index)
OH_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(brhousedata[object_cols]))
OH_cols.index=brhousedata.index

OH_brhousedata = pd.concat([brhousedata, OH_cols], axis=1)
OH_brhousedata.drop(object_cols, axis=1, inplace=True)
y1 = OH_brhousedata['total (R$)']

y2 = OH_brhousedata['rent amount (R$)']

X = OH_brhousedata.drop(['total (R$)', 'hoa (R$)', 'property tax (R$)', 'fire insurance (R$)', 'rent amount (R$)'], axis = 1)
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y2, random_state=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



model = RandomForestRegressor(random_state=1)

model.fit(X_train,y_train)



preds = model.predict(X_test)



score = mean_absolute_error(y_test,preds)
print(score)