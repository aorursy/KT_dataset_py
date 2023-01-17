# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

df.shape
df.head()
df_numeric = df.drop(['Overall rank', 'Country or region'],axis=1)

df_numeric.head()



# shuffle the DataFrame rows 

df_numeric = df_numeric.sample(frac = 1) 
corr_matrix = df_numeric.corr()
corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)
y = df_numeric['Score'].to_numpy()

X = df_numeric.drop(['Score'], axis=1).to_numpy()



#X = np.log(X+0.0001)



reg = LinearRegression().fit(X, y)

r_sq = reg.score(X, y)



y_hat = reg.predict(X)
plt.scatter(y_hat, y);

round(r_sq,2)
residuals = y - y_hat

plt.scatter(y_hat, residuals);