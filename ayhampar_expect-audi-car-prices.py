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
import pandas as pd 
data=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/audi.csv") 
data.head()
data.shape
data.isnull().sum()
data['model'].value_counts()
data=data[data['model'].map(data['model'].value_counts()) > 100]

data.shape
data.corr()

data.head()
data.drop(columns=['transmission','fuelType'],inplace=True)
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data['model'] = LE.fit_transform(data['model'])
data['year'] = LE.fit_transform(data['year'])

labele=data['price']
features=data.drop(columns=['price'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labele, test_size=0.2,random_state=5 )
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
pred=reg.predict(X_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,pred)
print(r2)