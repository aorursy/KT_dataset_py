# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/sao-paulo-real-estate-sale-rent-april-2019/sao-paulo-properties-april-2019.csv')
data.head()
data.isnull().sum()
data.shape
data.dtypes
# Checking unique values for negotition types

data['Negotiation Type'].unique(), data['Property Type'].unique()
# We have a good proportions of values between our dataset, this could be nice for us to 

#maybe predict which appartment is on rent or sale,

#or maybe create a different model for prices according to the negotiation type

data['Negotiation Type'].value_counts()
# to split data we have to distinguish wich columns we want to split, in our case im splitting District and city

data[['District','City']] = data['District'].str.split('/', expand = True)
data.head()
data.describe()
## So we can drop the last column City from our dataset

data = data.drop(columns = 'City', axis = 1)

data.head()
plt.figure(figsize = (12,8))

ax = sns.countplot(x = 'Negotiation Type', data = data, alpha = 0.8)

ax.set_title('Amount of apartaments for sale and in rent in São Paulo', fontsize = 22)

ax.set_ylabel('Number of apartments', fontsize = 1)

ax.set_xlabel('Status', fontsize = 16)
data['District'].value_counts()
ax = sns.catplot(x = 'Negotiation Type', y = 'Price', kind = 'boxen',

                data = data)
sns.scatterplot(x = 'Condo', y = 'Price', hue = 'Negotiation Type', data = data)
# Let's keep exploring out data

plt.figure(figsize = (12,8))

ax = sns.catplot(x = 'Suites', y = 'Price',data = data[data['Negotiation Type'] == 'sale'], kind = 'boxen');
plt.figure(figsize = (10,8))

ax = sns.regplot(x = 'Size', y = 'Price', data = data[data['Negotiation Type'] == 'sale'])
plt.figure(figsize = (10,6))

ax = sns.scatterplot(x = 'Size', y = 'Condo', hue = 'Negotiation Type', data = data)



ax.set_title('Condo x Size', fontsize = 22)

ax.set_ylabel('Condo', fontsize = 16)

ax.set_xlabel('Size', fontsize = 16)
sns.relplot(x = 'Size', y = 'Condo', data = data, hue = 'Negotiation Type', col = 'Negotiation Type')
data_rent = data[data['Negotiation Type'] == 'rent']

data_sale = data[data['Negotiation Type'] == 'sale']
# Let's keep exploring out data

ax = sns.catplot(x = 'Rooms', y = 'Price',data = data_sale, kind = 'boxen',

            palette = 'cubehelix')
plt.figure(figsize =(10,6))

ax = sns.boxplot(x = 'Toilets', y = 'Price', data = data_sale,

                palette = 'plasma')



ax.set_title('Distribution of toilets', fontsize = 22)

ax.set_ylabel('Price', fontsize = 16)

ax.set_xlabel('Size', fontsize = 16)

grouped = data_sale.groupby('District')['Price'].mean().reset_index()

grouped = grouped.sort_values(by = 'Price',ascending = False)

plt.figure(figsize = (40,40))

sns.set(font_scale = 2.3)

ax = sns.barplot(x='Price', y='District', data= grouped,palette = 'plasma')
sns.set(font_scale = 1)
data_rent.describe()
# Let's keep exploring out data

plt.figure(figsize =(10,8))

ax = sns.boxplot(x = 'Rooms', y = 'Price',data = data_rent,

            palette = 'cubehelix')
# Let's keep exploring out data

ax = sns.catplot(x = 'Suites', y = 'Price',data = data_rent, kind = 'boxen',

            palette = 'cubehelix')
grouped = data_rent.groupby('District')['Price'].mean().reset_index()

grouped = grouped.sort_values(by = 'Price',ascending = False)

plt.figure(figsize = (40,40))

sns.set(font_scale = 2.3)

ax = sns.barplot(x='Price', y='District', data= grouped,

                palette = 'plasma')

ax.set_title('Mean Price per District', fontsize = 45)
sns.set(font_scale = 1)
sns.catplot(x='Parking', y='Price', data=data_rent);

plt.figure(figsize =(10,8))

ax = sns.boxplot(x = 'Furnished', y = 'Price',data = data_rent,

            palette = 'cubehelix')
plt.figure(figsize=(10,10))

ax = sns.distplot(data_rent['Price'])
ax = sns.pairplot(data_rent, y_vars = 'Price', x_vars = ['Condo', 'Size', 'Rooms','Toilets','Suites','Parking','Elevator'], height = 5, kind = 'reg')

ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)

ax
data_rent['log_Price'] = np.log(data_rent['Price'])

data_rent['log_Condo'] = np.log(data_rent['Condo'])

data_rent['log_Size'] = np.log(data_rent['Size'])

data_rent['log_Furnished'] = np.log(data_rent['Furnished']+1)

data_rent['log_Swim'] = np.log(data_rent['Swimming Pool']+1)

data_rent['log_Elevator'] = np.log(data_rent['Elevator']+1)
data_rent.head()
# correlation plot

plt.figure(figsize = (10,8))

corr = data_rent.corr()

sns.heatmap(corr, cmap = 'plasma', annot= True);
y = data_rent['log_Price']
X = data_rent[['log_Size', 'log_Swim', 'log_Furnished']]
## Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
import statsmodels.api as sm
X_train_const = sm.add_constant(X_train)

X_train
X_train_const
model_statsmodels = sm.OLS(y_train,X_train_const, hasconst = True).fit()
model_statsmodels.summary()
print(model_statsmodels.summary())
from sklearn.linear_model import LinearRegression

from sklearn import metrics
model = LinearRegression()

model.fit(X_train,y_train)
model.score(X_train,y_train).round(3)
y_pred = model.predict(X_test)
metrics.r2_score(y_test, y_pred).round(3)
print(model.coef_)

print(model.intercept_)
X_train.columns
index = ['Intercept','log_Size', 'log_Swim', 'log_Furnished']
pd.DataFrame(data= np.append(model.intercept_, model.coef_), index = index, columns = ['Parameters'])
y_pred_train = model.predict(X_train)
ax = sns.scatterplot(x = y_pred_train, y = y_train)

ax.figure.set_size_inches(12, 6)

ax.set_title('Pred X Real', fontsize=18)

ax.set_xlabel('log Price - Pred', fontsize=14)

ax.set_ylabel('log Price - Real', fontsize=14)

ax
## Getting residuals

resid = y_train - y_pred_train
## Ploting the histogram of the residuals

plt.figure(figsize=(10,8))

sns.distplot(resid)

ax.figure.set_size_inches(20, 10)

ax.set_title('Histogram from our resid', fontsize=18)

ax.set_xlabel('log Price', fontsize=14)

ax