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
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv") 
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv") 
train = train.copy()
test = test.copy()
train.head()


test.head()
print("Test Data Shape",test.shape)
print("Train Data Shape", train.shape)
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)
print(train.SalePrice.describe())




print("Skew is", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()
print(train.OverallQual.unique())
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.mean)
print(quality_pivot)
quality_pivot.plot(kind='bar', color='green')
plt.xlabel('Overall Quality')
plt.ylabel('Mean Sale Price')
plt.xticks(rotation=0)
plt.show()
target = np.log(train.SalePrice)
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
train = train[train['GarageArea'] < 1200]

#display the previous graph again without outliers
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
#nulls
print(nulls)
print ("Unique values are:", train.MiscFeature.unique())
#consider the non-numeric features and display details of columns
categoricals = train.select_dtypes(exclude=[np.number])
#categoricals.describe()
print(categoricals.describe())