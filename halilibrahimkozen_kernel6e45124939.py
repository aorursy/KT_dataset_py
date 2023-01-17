# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)
train.head()
train1 = train.fillna(train.mean())
train1.head()
df1 = train1.select_dtypes(include=['object','float64','int64']).copy()
df1.head(10)
df1 = df1.apply(lambda x:x.fillna(x.value_counts().index[0]))
df1
corrmat = train.corr()
corr_features = corrmat.index
plt.figure(figsize=(40,40))
sns.heatmap(train[corr_features].corr(),annot=True,cmap=plt.cm.Reds)
corrmat['SalePrice'].head()
corrmat['SalePrice']
corr_dict=corrmat['SalePrice'].sort_values(ascending=False).to_dict()
my_columns=[]
for key,value in corr_dict.items():
    if ((value>0.2) & (value<0.8)) | (value<=-0.1):
        my_columns.append(key)
        
df1[my_columns]
X = df1.iloc[:,1:80] 
y = df1.iloc[:,-1] 

train_NaN=pd.DataFrame(X[my_columns].isnull().sum(),columns=['Number of NaN'])
train_NaN
test=pd.read_csv('test.csv')
test.head()
corre=train.corr()
plt.figure(figsize=(3,8))
sb.heatmap(corre[['SalePrice']].sort_values(by=['SalePrice'],ascending=True).tail(10),vmin=-1, cmap='Reds', annot=True)
#X = train.iloc[:,1:80] 
#y = train.iloc[:,-1] 
X=df1.drop(['SalePrice'],axis=1)
y=df1.SalePrice
df2 = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
df2.head()
N=pd.DataFrame(df2[my_columns].isnull().sum(),columns=['Number of NaN'])
N
X_train = X[my_columns]
y_train = train['SalePrice']
X_test = test[my_columns]
X_test
lm = LinearRegression()
lm.fit(X_train,y_train)
#y_=lm.predict(X_test)
#print('R square Accuracy: ',r2_score(y_test,y_))