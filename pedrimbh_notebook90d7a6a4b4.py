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
house_df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

house_df.head()
house_df.drop('id',axis=1, inplace=True)

house_df.drop('date',axis=1, inplace=True)

house_df.drop(['lat','long'],axis=1, inplace=True)
house_df.head()
house_df.corr().round(2)
house_df.drop('zipcode',axis=1, inplace=True)
house_df.describe().round(2)
len(house_df)
house_df.isna().sum()
import seaborn as sns
sns.set_palette("GnBu_r")

ax = sns.pairplot(house_df, y_vars='price', x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront','view','condition',

                             'grade','sqft_above','yr_built','yr_renovated','sqft_living15','sqft_lot15'])

ax.fig.suptitle('Dispersão', fontsize= 20, y=1.1);
house_df.query('bedrooms > 20')
house_df.drop(15870,inplace=True)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
X = house_df.drop('price', axis=1)
y= house_df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model.fit(X_train,y_train)

result = model.score(X_test,y_test) 

print(result.round(4))
datatest = X_test[0:20]

model.predict(datatest).round(2)
y_test[0:20]