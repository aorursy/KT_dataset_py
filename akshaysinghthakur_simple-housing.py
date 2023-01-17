# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as p # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = p.read_csv("/kaggle/input/Housing.csv")
a = dataset.iloc[:,:-1].values
b = dataset.iloc[:,3].values
from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test = train_test_split(a,b ,test_size=0.1, random_state=0)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(a_train,b_train)
b_pred = regr.predict(a_test)
b_pred
b_test
