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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.pandas.set_option('display.max_columns',None)
dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dataset.shape
dataset.head()
featuresWithNa = [features for features in dataset.columns if dataset[features].isnull().sum()>1]

#print 
for feature in featuresWithNa:
    print(feature, np.round(dataset[feature].isnull().mean(), 4), '%missing values')
for feature in featuresWithNa:
    data = dataset.copy()
    
    #it will assign 1 if null and 0 if not
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
data.info()
print("Id of house {}".format(len(dataset.Id)))
