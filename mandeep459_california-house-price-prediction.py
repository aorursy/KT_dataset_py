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
bd=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
bd.head()
bd.info()
bd.describe()
import matplotlib.pyplot as plt
%matplotlib inline
bd.hist(bins=50,figsize=(20,15)) 
plt.show()
bd.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
             s=bd["population"]/100,label="population",figsize=(10,7),
             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True, 
            )
plt.legend()
corr_matrix=bd.corr()
corr_matrix["median_house_value"].sort_values(ascending=False) 
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(bd.corr(), annot=True)
from pandas.plotting import scatter_matrix
attributes = ["median_house_value",	"median_income","total_rooms","housing_median_age"] 
scatter_matrix(bd[attributes],figsize=(12,8))

sns.distplot(bd.median_house_value)
bd.isnull().sum()
df= bd.fillna(bd.mean())
## convert ocean_proximity variable  into the number 
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
print(df["ocean_proximity"].value_counts())
df["ocean_proximity"] = labelEncoder.fit_transform(df["ocean_proximity"])
df["ocean_proximity"].value_counts()
df.describe()
df1 = df.drop(['median_house_value'], axis=1)
df2 = df.median_house_value

from sklearn.model_selection import train_test_split
df1_train, df1_test,df2_train, df2_test = train_test_split(df1,df2, test_size=0.2, random_state=19)    
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
linear_reg = linear_model.LinearRegression()
linear_reg.fit(df1_train,df2_train)
r2_score(linear_reg.predict(df1_train),df2_train)
# Final predictions
kd=linear_reg.predict(df1_test)
kd
