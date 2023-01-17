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
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
missing_count= (df.isnull().sum()/len(df))*100
missing_count= missing_count[missing_count>0]
print(missing_count.sort_values())
df=df.drop(['Fence','Alley','MiscFeature','PoolQC'],axis=1)

df.shape
numeric_data = df.select_dtypes(include=[np.number])
categorical_data=df.select_dtypes(exclude=[np.number])
print('there are {0} numerical and{1} categorical in the data set'. 
      format(numeric_data.shape[1],categorical_data.shape[1]))
df.groupby(['MSSubClass']).Id.count().plot(kind='bar',figsize=(14,4))
df.groupby(['YrSold','MoSold']).Id.count().plot(kind='bar',figsize=(14,4))
df['YrSold']=df['YrSold'].astype(object)
df['MSSubClass']=df['MSSubClass'].astype(object)
df['MoSold']=df['MoSold'].astype(object)
numeric_data = df.select_dtypes(include=[np.number])
categorical_data=df.select_dtypes(exclude=[np.number])
print('there are {0} numerical and{1} categorical in the data set'. 
      format(numeric_data.shape[1],categorical_data.shape[1]))
numeric_data.isnull().sum()
numeric_data.info()
numeric_data['LotFrontage']=numeric_data.LotFrontage.fillna(numeric_data.LotFrontage.mean())
numeric_data['MasVnrArea']=numeric_data.MasVnrArea.fillna(numeric_data.MasVnrArea.mean())
numeric_data['GarageYrBlt']=numeric_data.GarageYrBlt.fillna(numeric_data.GarageYrBlt.mean())
numeric_data.shape
numeric_data=numeric_data.drop('SalePrice',axis=1)
numeric_data.shape
numeric_data['GarageYrBlt']=numeric_data['GarageYrBlt'].astype(int)
numeric_data['LotFrontage']=numeric_data['LotFrontage'].astype(int)
numeric_data['MasVnrArea']=numeric_data['MasVnrArea'].astype(int)
categorical_data.shape
categorical_data.isna().sum()
a_cat=categorical_data.fillna(method='bfill')
a_cat=a_cat.fillna(method='ffill')
a_cat['YrSold'].unique()
a_cat['MoSold'].unique()
a_cat['MSSubClass'].unique()
a_cat['YrSold']=a_cat['YrSold'].astype(object)
a_cat['MoSold']=a_cat['MoSold'].astype(object)
a_cat['MSSubClass']=a_cat['MSSubClass'].astype(object)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
for col in a_cat:
    if a_cat[col].dtypes=='object':
        a_cat[col]=le.fit_transform(a_cat[col])
a_cat.shape
a_cat.columns #2,5,28(only having [0,1])[names:street,utilities,centralAir]
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()

a_cat.shape
a_cat_array=a_cat.iloc[:,0:42].values
a_cat_array
cols=[0,1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41]

a_cat_onehot1=one.fit_transform(a_cat_array[:,cols])
a_cat_onehot1.shape
a_cat_onehot1_table=pd.DataFrame(data=a_cat_onehot1.toarray())
a_cat_onehot1_table.shape
complete_table=pd.concat([numeric_data,a_cat_onehot1_table],axis=1)
complete_table.shape
X_train=complete_table
Y_train=df['SalePrice']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=1,shuffle=True)
x_train.shape,x_test.shape
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
pred=dt.predict(x_test)
dt.score(x_test,y_test)
dt.feature_importances_
df_withoutliers=pd.DataFrame({'actual':y_test,'predicted':pred})
df_withoutliers['diff'] = abs(df_withoutliers['actual'] - df_withoutliers['predicted']) 
df_withoutliers['mape'] = df_withoutliers['diff']/df_withoutliers['actual']           
df_withoutliers['ID']=df_withoutliers.index
sns.boxplot(data=y_train)
y_train[y_train<299000]
sns.boxplot(data=y_train[y_train<299000])
plt.show()
x_train=x_train[y_train<299000]
x_train.shape
y_train=y_train[y_train<299000]
y_train.shape
from sklearn.tree import DecisionTreeRegressor
dt1=DecisionTreeRegressor()
dt1.fit(x_train,y_train)
predict=dt1.predict(x_test)
dt1.score(x_test,y_test)
dt.score(x_test,y_test)
dt.feature_importances_
# checking feature importance in the model with outliers
dt1.feature_importances_ 
# checking feature importance in the model without outliers
difference=pd.read_csv('../input/feature-importances-sheet/finalresult(17E).csv')
difference.head()
x_train['OverallQual'].value_counts() #with outliers data
x_train['OverallQual'].value_counts()   #without outliers data
x_test['OverallQual'].value_counts()
x_test[x_test['OverallQual']>8]
df1_without_outliers=pd.DataFrame({'actual':y_test,'predicted':predict})
df1_without_outliers.shape
df1_without_outliers['diff'] = abs(df1_without_outliers['actual'] - df1_without_outliers['predicted']) 
df1_without_outliers['mape'] = df1_without_outliers['diff']/df1_without_outliers['actual']
df1_without_outliers['ID']=df1_without_outliers.index
df_without_outliers_OveralQualityof9and10=df1_without_outliers[df1_without_outliers['ID'].isin([53,994,798,309,1373,644,1169,1228,885,1182,1036,481,724,58,765,336,350,527,1267,1338])]
df_without_outliers_removed9and10=df1_without_outliers.drop([53,994,798,309,1373,644,1169,1228,885,1182,1036,481,724,58,765,336,350,527,1267,1338])
df_without_outliers_removed9and10.shape
df_withoutliers_overallquality9and10=df_withoutliers[df_withoutliers['ID'].isin([53,994,798,309,1373,644,1169,1228,885,1182,1036,481,724,58,765,336,350,527,1267,1338])] #overall quality
df_withoutliers_removing9and10 = df_withoutliers.drop([53,994,798,309,1373,644,1169,1228,885,1182,1036,481,724,58,765,336,350,527,1267,1338])
df_withoutliers['mape'].describe()   #total data
df1_without_outliers['mape'].describe()  #total data
df_withoutliers_removing9and10['mape'].describe()   #REMOVING THE DATA OF 9,10
df_without_outliers_removed9and10['mape'].describe()   #REMOVING THE DATA OF 9,10
df_withoutliers_overallquality9and10['mape'].describe() #ONLY THE DATA OF 9,10
df_without_outliers_OveralQualityof9and10['mape'].describe() #ONLY THE DATA OF 9,10
