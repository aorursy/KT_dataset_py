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
!pip install pycaret
data = pd.read_csv('/kaggle/input/housedata/Bengaluru_House_Data.csv')
data.head()

df = data.iloc[:100]
df.shape
df.info()
df.drop(df.loc[df['total_sqft']=='2957 - 3450'].index, inplace=True)

# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = df, target = 'price',numeric_features=['total_sqft','bath','balcony'])
# return best model
best = compare_models()
top3 = compare_models(n_select = 3)
huber = create_model('huber')
huber = tune_model(huber)
plot_model(huber,plot='error')
plot_model(huber,plot='learning')