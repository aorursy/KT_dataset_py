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
house_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
house_train
house_train.SalePrice.describe()
import matplotlib.pyplot as plt
import numpy as np
target = np.log(house_train.SalePrice)
plt.hist(target)
plt.show()
house_train.OverallQual.unique()
plt.scatter(x=house_train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
house_train = house_train[house_train['GarageArea'] < 1200]
categoricals = house_train.select_dtypes(exclude=[np.number])
categoricals.describe()
house_train['enc_street'] = pd.get_dummies(house_train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(house_train.Street, drop_first=True)
nulls = pd.DataFrame(house_train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls