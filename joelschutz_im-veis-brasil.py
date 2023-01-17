# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as venus

venus.set_style('whitegrid')

from sklearn.model_selection import train_test_split

import statsmodels.api as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head(50)
data.describe().round(3)
data['furniture'] = data['furniture'].apply(lambda x: 1 if x == 'furnished' else 0)

data['condomine'] = data['hoa (R$)'].apply(lambda x: 1 if x > 0 else 0)

data['floor'] = data['floor'].apply(lambda x: 0 if not x.isnumeric() else int(x))

data['animal'] = data['animal'].apply(lambda x: 1 if x == 'acept' else 0)
data['city'].unique()
selection = data['city'] == 'São Paulo'

data_sao_paulo = data[selection]

data_sao_paulo.drop(columns=['city'], inplace=True)

venus.set_palette('RdYlGn')

data_sao_paulo
data_sao_paulo.corr().round(3)
ax = venus.distplot(data_sao_paulo['rent amount (R$)'])

ax.figure.set_size_inches(20, 6)
data_sao_paulo.drop(data_sao_paulo[data_sao_paulo['property tax (R$)']>300000].index, inplace=True)

data_sao_paulo.drop(data_sao_paulo[data_sao_paulo['area']>2000].index, inplace=True)
ax = venus.distplot(data_sao_paulo['property tax (R$)'])

ax.figure.set_size_inches(20, 6)
analysis_data = pd.DataFrame(columns=['rent','area','rooms','bathrooms','parking spaces','animals','furniture'])

analysis_data['rent'] = np.log(data_sao_paulo['rent amount (R$)'])

analysis_data['area'] = np.log(data_sao_paulo['area'])

analysis_data['rooms'] = np.log(data_sao_paulo['rooms'])

analysis_data['bathrooms'] = np.log(data_sao_paulo['bathroom'])

analysis_data['parking spaces'] = np.log(data_sao_paulo['parking spaces']+1)

analysis_data['animals'] = np.log(data_sao_paulo['animal']+1)

analysis_data['furniture'] = np.log(data_sao_paulo['furniture']+1)

analysis_data
ax = venus.distplot(analysis_data['rent'])

ax.figure.set_size_inches(20, 6)
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='animals')
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='furniture')
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','parking spaces','animals','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_sao_paulo = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_sao_paulo.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_sao_paulo.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','parking spaces','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_sao_paulo = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_sao_paulo.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_sao_paulo.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
selection = data['city'] == 'Porto Alegre'

data_porto_alegre = data[selection]

data_porto_alegre.drop(columns=['city'], inplace=True)

venus.set_palette('Blues_r')

data_porto_alegre
data_porto_alegre.describe().round(3)
data_porto_alegre.corr().round(3)
data_porto_alegre.drop(data_porto_alegre[(data_porto_alegre['area']>700) & (data_porto_alegre['rent amount (R$)']<2000)].index, inplace=True)
ax = venus.distplot(data_porto_alegre['rent amount (R$)'])

ax.figure.set_size_inches(20, 6)
analysis_data = pd.DataFrame(columns=['rent','area','rooms','bathrooms','parking spaces','animals','furniture'])

analysis_data['rent'] = np.log(data_porto_alegre['rent amount (R$)'])

analysis_data['area'] = np.log(data_porto_alegre['area'])

analysis_data['rooms'] = np.log(data_porto_alegre['rooms'])

analysis_data['bathrooms'] = np.log(data_porto_alegre['bathroom'])

analysis_data['parking spaces'] = np.log(data_porto_alegre['parking spaces']+1)

analysis_data['animals'] = np.log(data_porto_alegre['animal']+1)

analysis_data['furniture'] = np.log(data_porto_alegre['furniture']+1)

analysis_data
ax = venus.distplot(analysis_data['rent'])

ax.figure.set_size_inches(20, 6)
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='animals')
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='furniture')
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','parking spaces','animals','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_porto_alegre = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_porto_alegre.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_porto_alegre.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
y = analysis_data['rent']

X = analysis_data[['area','bathrooms','parking spaces','animals','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_porto_alegre = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_porto_alegre.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_porto_alegre.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
selection = data['city'] == 'Rio de Janeiro'

data_rio_de_janeiro = data[selection]

data_rio_de_janeiro.drop(columns=['city'], inplace=True)

venus.set_palette('Greens_r')

data_rio_de_janeiro
data_rio_de_janeiro.describe().round(3)
data_rio_de_janeiro.corr().round(3)
ax = venus.distplot(data_rio_de_janeiro['rent amount (R$)'])

ax.figure.set_size_inches(20, 6)
analysis_data = pd.DataFrame(columns=['rent','area','rooms','bathrooms','parking spaces','animals','furniture'])

analysis_data['rent'] = np.log(data_rio_de_janeiro['rent amount (R$)'])

analysis_data['area'] = np.log(data_rio_de_janeiro['area'])

analysis_data['rooms'] = np.log(data_rio_de_janeiro['rooms'])

analysis_data['bathrooms'] = np.log(data_rio_de_janeiro['bathroom'])

analysis_data['parking spaces'] = np.log(data_rio_de_janeiro['parking spaces']+1)

analysis_data['animals'] = np.log(data_rio_de_janeiro['animal']+1)

analysis_data['furniture'] = np.log(data_rio_de_janeiro['furniture']+1)

analysis_data
ax = venus.distplot(analysis_data['rent'])

ax.figure.set_size_inches(20, 6)
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='animals')
ax = venus.pairplot(analysis_data, y_vars='rent', x_vars=['area','rooms','bathrooms','parking spaces'], height=5, kind='reg', hue='furniture')
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','parking spaces','animals','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_rio_de_janeiro = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_rio_de_janeiro.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_rio_de_janeiro.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','parking spaces','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_rio_de_janeiro = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_rio_de_janeiro.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_rio_de_janeiro.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)
y = analysis_data['rent']

X = analysis_data[['area','rooms','bathrooms','furniture']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train_w_constant = sm.add_constant(X_train)

model_rio_de_janeiro = sm.OLS(y_train, X_train_w_constant, hasconst = True).fit()

print(model_rio_de_janeiro.summary())
X_test_w_constant = sm.add_constant(X_test)

y_predi = model_rio_de_janeiro.predict(X_test_w_constant)

ax = venus.scatterplot(x=y_predi, y=y_test)