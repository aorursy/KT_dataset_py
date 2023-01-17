# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(df_train['SalePrice'])

# Plot fig sizing. 

# style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(df_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(df_train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
missing_value = df_train.isnull().sum(axis = 0)

percent = (df_train.isnull().sum(axis = 0)/df_train.isnull().count(axis = 0)) * 100

labels = df_train.columns 

summary_na = pd.DataFrame({'name' : labels, 'missing_values' : missing_value, 'percent_missing' : percent})

summary_na.sort_values(by=['percent_missing'], ascending=False).head(20)
df_train2 = df_train.drop(columns = [summary_na['name'][x] for x in range(len(summary_na)) if summary_na['percent_missing'][x] > 15])
X = df_train2.iloc[:,0:79]

y = df_train2.iloc[:,80].values
from sklearn.impute import SimpleImputer



imp = SimpleImputer(strategy="most_frequent")

X = pd.get_dummies(X)

X = imp.fit_transform(X)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler_train = scaler.fit_transform(X)

scaler_train
from sklearn.linear_model import LinearRegression



reg = LinearRegression().fit(scaler_train, y)

print("Accuracy :", reg.score(scaler_train, y))

y_pred = reg.predict(scaler_train)



from sklearn.metrics import r2_score, mean_squared_error

print("R2 :" ,r2_score(y, y_pred))

print("RMSE :", mean_squared_error(y, y_pred))
result = pd.DataFrame({'y_true' : y, 'y_pred' : y_pred})

print(result)

plt.subplots(figsize=(12,9))

plt.plot(result)