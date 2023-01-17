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
import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
data =  pd.read_csv('../input/dc-residential-properties/DC_Properties.csv')

data = data[['BEDRM', 'BATHRM', 'HF_BATHRM', 'GBA']]

data
data = data.dropna(subset=['GBA'])

data.isnull().any()

data = data.fillna(method='ffill')

data
X = data[['BEDRM', 'BATHRM', 'HF_BATHRM']].values

y = data['GBA'].values
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(data['GBA'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, ['BEDRM', 'BATHRM', 'HF_BATHRM'], columns=['Coefficient'])

coeff_df
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})



df1 = df.head(30)

df1
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))