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
# import libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



# bring the dataset
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# check the dataset
df_train.head()
df_train.columns
# descriptive statistics summary
df_train['SalePrice'].describe()
# histogram
sns.distplot(df_train['SalePrice']);
plt.xticks(rotation=45)
# skewness nd kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# scatter plot grlivarea/saleprice

data=pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis=1)

sns.scatterplot(x='GrLivArea',y='SalePrice',data=data)
#scatter plot totalbsmtsf/saleprice

var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
