import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.options.display.max_columns = None
df = pd.read_csv('/kaggle/input/apartments-in-volgograd-june-2020/apartments_in_volgograd_june_2020.csv', sep='\t', index_col=0)
df = df[~df['district'].isnull()]
df['district_1'] = (df['district'] == 'р-н Центральный').astype(int)

df['district_2'] = (df['district'] == 'р-н Советский').astype(int)

df['district_3'] = (df['district'] == 'р-н Дзержинский').astype(int)

df['district_4'] = (df['district'] == 'р-н Ворошиловский').astype(int)

df['district_5'] = (df['district'] == 'р-н Кировский').astype(int)

df['district_6'] = (df['district'] == 'р-н Тракторозаводский').astype(int)

df['district_7'] = (df['district'] == 'р-н Краснооктябрьский').astype(int)

df['district_8'] = (df['district'] == 'р-н Красноармейский').astype(int)

df = df.drop(['district'], axis=1)
X = df.drop(['price', 'address'], axis=1)

y = df['price']



X_train = X[:-20]

X_test = X[-20:]



y_train = y[:-20]

y_test = y[-20:]
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)



print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))

print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))
plt.scatter(X_train['area'], y_train,  color='black', alpha=0.3)

plt.show()