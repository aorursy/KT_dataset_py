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

import warnings

warnings.filterwarnings('ignore')

sns.set()
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
des=pd.read_fwf('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt')
des
print(train.shape)



print('-'*50)



print(test.shape)
sns.distplot(train['SalePrice'])
plt.hist(train['SalePrice'])
train['SalePrice'].skew()
# We can see that the prices are positively skewed in which case we can go for the log transformation.

#Also the predicted values will also need to be converted back.

target=np.log(train.SalePrice)

plt.hist(target,color='red')

plt.show()
target.skew()
# A value closer to 0 means we have improved the skewness .

# we can see tha it now follows the normal distribution.
numeric_features=train.select_dtypes(include=(np.number))

corr=numeric_features.corr()



print(corr['SalePrice'].sort_values(ascending=False)[:5])

print(corr['SalePrice'].sort_values(ascending=False)[-5:])
train.dtypes
#Exploring the pdfs
plt.scatter(x=train['GarageArea'],y=target)

plt.xlabel('Garage Area')

plt.ylabel('Sale Price')

plt.show()
train['GarageArea'].count()
# we can see that many of the observations have zero values indicating that there is no Garage . 

#Also  we can see that there are some outliers .

#Lets remove some outlier for better analysis.

train=train[train['GarageArea']<1200]
plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice))

plt.xlabel('Garage Area')

plt.ylabel('Sale Price')

plt.show()
# we can see that outliers have been removed
plt.scatter(x=train['LotArea'],y=np.log(train.SalePrice))

plt.xlabel('LotArea')

plt.ylabel('Sale Price')

plt.show()
nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:30])

nulls.columns=['Null Counts']

nulls.index.name='Features'
nulls
train.describe(include='all')
print(train.Street.value_counts())
#encoding the non-numeric values



train['enc_street']=pd.get_dummies(train.Street,drop_first=True)

test['enc_street']=pd.get_dummies(test.Street,drop_first=True)
print(train.enc_street.value_counts())
print(train.Alley.value_counts())
condition_pivot=train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)

condition_pivot.plot(kind='bar',color='blue')

plt.xlabel('Sale condition')

plt.ylabel('Sale Price')

plt.xticks(rotation=90)

plt.show()
# We can see that partial sales condition has a significant median value.

#So lets encode it as 1 and the rest as 0
def encode(x):

    return 1 if x=='Partial' else 0

train['enc_conditon']=train.SaleCondition.apply(encode)

test['enc_conditon']=test.SaleCondition.apply(encode)
condition_pivot=train.pivot_table(index='enc_conditon',values='SalePrice',aggfunc=np.median)

condition_pivot.plot(kind='bar',color='blue')

plt.xlabel('enc_conditon')

plt.ylabel('Sale Price')

plt.xticks(rotation=90)

plt.show()
data=train.select_dtypes(include=[np.number]).interpolate().dropna()
data.head()
data.shape
data_nulls=pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:30])

data_nulls.columns=['Null Counts']

data_nulls.index.name='Features'
data_nulls
x=data.drop(['SalePrice','Id'],axis=1)

y=np.log(train.SalePrice)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.33)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

model=reg.fit(x_train,y_train)

reg.score(x_test,y_test)*100
#Lets see how it works with the test data
y_pred=reg.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print('RMSE is:',mean_squared_error(y_test,y_pred))
print('R-squared is:',r2_score(y_test,y_pred))
actual_value=y_test

predictions=y_pred
plt.scatter(x=actual_value,y=predictions,color='red')

plt.title('Regression model')

plt.xlabel('actual_values')

plt.ylabel('Predictions')

plt.show()
submission=pd.DataFrame()

submission['id']=test.Id
# select the features from the test data for the model.

feat=test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
prediction=model.predict(feat)
# remember how we convert  log we we have to reverse the transformation now 



final_prediction=np.exp(prediction)
print('original predictions are:',prediction)

print('Final predictions are:',final_prediction)
submission['SalePrice']=final_prediction
submission.head()
submission.to_csv('submission.csv',index=False)