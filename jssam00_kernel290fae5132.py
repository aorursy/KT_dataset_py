# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head(10)
df.dtypes
df = df.drop('id',axis=1)
z  = df['bedrooms']
z.replace("0", np.nan, inplace = True)
df.dropna(axis=0, inplace=True)
df.describe()
y = df["floors"].value_counts()
y.to_frame()
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x="waterfront", y = "price" , data = df )
sns.regplot(x="sqft_above", y="price", data=df)
plt.ylim(0,)
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression as lr
x = df[['sqft_living']]
y = df[['price']]
lm = lr()
lm.fit(x,y)
lm.score(x,y)
z = df[["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm.fit(z,y)
lm.score(z,y)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures as pr
from sklearn.preprocessing import StandardScaler
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',lr())]
pipe = Pipeline(Input)
pipe.fit(z,y)
pipe.score(z,y)
ridge = Ridge(alpha = 0.1)
ridge.fit(z,y)
ridge.score(z,y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures as pr
ps = pr(degree = 2)
z_train ,z_test,y_train,y_test = train_test_split(z,y,test_size = 0.3 , random_state = 0)
z_train_pr = ps.fit_transform(z_train)
z_test_pr = ps.fit_transform(z_test)
lm.fit(z_train_pr,y_train)
lm.score(z_train_pr,y_train)

lm.fit(z_test_pr,y_test)
lm.score(z_test_pr,y_test)
ridge = Ridge(alpha = 0.1)
ridge.fit(z_train_pr,y_train)
ridge.score(z_train_pr,y_train)
