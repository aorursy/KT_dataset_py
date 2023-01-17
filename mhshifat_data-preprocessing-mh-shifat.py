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
dataset = pd.read_csv('/kaggle/input/buy-product-based-on-county-age-and-salary/Data.csv')
dataset
x = dataset.iloc[:,:-1].values 
x
y = dataset.iloc[:,3].values
y
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy = "mean")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
x
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
x
transformer = ColumnTransformer(
    transformers = [
        ("OneHot",
         OneHotEncoder(),
         [0]
        )
    ],
    remainder='passthrough'
)
x = transformer.fit_transform(x)
x
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0) #'0.2' means 20%
x_train
y_train
x_test
x_train
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
x_train
x_test
