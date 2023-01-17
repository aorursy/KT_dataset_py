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

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split





dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

dataset
dataset.info()
cols_with_missing = [col for col in dataset.columns 

                                 if dataset[col].isnull().any()]

reduced_original_data = dataset.drop(cols_with_missing, axis=1)

reduced_test_data = test_data.drop(cols_with_missing, axis=1)

reduced_original_data
reduced_original_data.info()
for x in reduced_original_data:

    if (reduced_original_data[x].dtype =='O'):

        reduced_original_data[x] = reduced_original_data[x].astype('category')

        reduced_original_data[x] = reduced_original_data[x].cat.codes
reduced_original_data.head()
reduced_original_data.info()
for x in reduced_test_data:

    if (reduced_test_data[x].dtype =='O'):

        reduced_test_data[x] = reduced_test_data[x].astype('category')

        reduced_test_data[x] = reduced_test_data[x].cat.codes
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

reduced_test_data_with_imputed_values = my_imputer.fit_transform(reduced_test_data)
for col in reduced_test_data.columns:

    if reduced_test_data[col].isnull().any():

        print (col)

        reduced_test_data[col] = reduced_test_data[col].fillna(0).astype(np.int64)

        print (reduced_test_data[col].dtype)
reduced_test_data.isnull().any().sum()
X = reduced_original_data.drop('SalePrice' , axis =1 )

y = reduced_original_data['SalePrice']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0. , random_state=0)



reg = LinearRegression()



reg.fit(X , y)

reg.score(X , y)
# reg.score(X_test , y_test)
for i in range (0 , len(reduced_test_data)):

    print (reduced_test_data.Id[i] , reg.predict (reduced_test_data.loc[reduced_test_data.index[i]].values.reshape(1,-1)) )
d = {'Id': reduced_test_data.Id, 'SalePrice': reg.predict (reduced_test_data)}

d
submission_frame = pd.DataFrame (d)

submission_frame
submission_frame.to_csv ('submission_frame.csv',index=False)