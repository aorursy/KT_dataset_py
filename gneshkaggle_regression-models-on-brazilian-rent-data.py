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
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head()
data=data.rename(columns={'hoa (R$)':'hoa',
                           'rent amount (R$)':'rent amount',
                           'property tax (R$)':'property tax',
                           'fire insurance (R$)':'fireinsurance',
                           'total (R$)':'total'})
data.head()
data.info()
data['floor'].unique()
data['floor']=data['floor'].replace('-',0)
data['floor']=data['floor'].astype('int64')
data.describe()
sns.boxplot('rent amount',data=data);
#plot city vs rentamount
sns.boxplot(x='city',y='rent amount',data=data);
corr=data.corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr,annot=True,fmt='.2f',cmap=plt.cm.Blues);
#plot of rooms vs rentamount
plt.subplot(2,1,1)
sns.boxplot(data['rooms'])

plt.subplot(2,1,2)
sns.barplot(x='rooms',y='rent amount',data=data);
#parking spaces vs rent amount
plt.subplot(2,1,1)
sns.boxplot(data['parking spaces'])

plt.subplot(2,1,2)
sns.barplot(x='parking spaces',y='rent amount',data=data);
#plot of rooms vs rentamount
plt.subplot(2,1,1)
sns.boxplot(data['bathroom'])

plt.subplot(2,1,2)
sns.barplot(x='bathroom',y='rent amount',data=data);
#furniture 
sns.countplot(x='furniture',data=data);
#animal
sns.countplot(x='animal',data=data);
#plot of rooms vs rentamount
plt.subplot(2,1,1)
sns.boxplot(x='rent amount',data=data[data['animal']=='acept']);

plt.subplot(2,1,2)
sns.boxplot(x='rent amount',data=data[data['animal']=='not acept']);
#fireinsurance vs rent amount
sns.scatterplot(x='rent amount',y='fireinsurance',data=data);
city_group=data.groupby('city')['rent amount']
Q1 = city_group.quantile(.25)
Q3 = city_group.quantile(.75)

# IQR = Interquartile Range
IQR = Q3 - Q1

# Limits
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(lower)
print(upper)
new_data=pd.DataFrame()

for city in city_group.groups.keys():
    df_select=data[(data['city']==city)&(data['rent amount']>lower[city])&(data['rent amount']<upper[city])]
    new_data=pd.concat([new_data,df_select])
    
new_data.head()
    
#plot of rent amount after removing outliers
sns.boxplot(x='city',y='rent amount',data=new_data);
#removing the rows that contain single valued column as it may give an error while one hot encoding if there is no instance in training data with that value. 
new_data=new_data[new_data['parking spaces']!=10]
features=['city','rooms','bathroom','parking spaces','furniture','fireinsurance']
X=new_data[features]
y=new_data['rent amount']
X.head()
for col in X.columns[:-1]:
    X[col]=X[col].astype('category')
    
X['fireinsurance']=X['fireinsurance'].astype('int64')
X.info()
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,mean_absolute_error

catTransformer=Pipeline(steps=[('onehot',OneHotEncoder())])
numTransformer=Pipeline(steps=[('scaler',StandardScaler())])
numFeatures=X.select_dtypes(include=['int','float']).columns
numFeatures
catFeatures=X.select_dtypes(include=['category']).columns
catFeatures
preprocessor=ColumnTransformer(transformers=[('numeric',numTransformer,numFeatures),
                                             ('categoric',catTransformer,catFeatures)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)
regressors=[DecisionTreeRegressor(),
           LinearRegression(),
           SVR(), 
           RandomForestRegressor(),
           XGBRegressor()]
np.random.seed(123)

for regressor in regressors:
    
    estimator=Pipeline([('preprocessor',preprocessor),
                        ('regressor',regressor)])
    estimator.fit(X_train,y_train)
    preds=estimator.predict(X_test)
    
    print(regressor)
    print('Mean squared error: ',mean_squared_error(y_test,preds))
    print('mean_absolute_error: ',mean_absolute_error(y_test,preds))
    print('r2_score: ',r2_score(y_test,preds))
    print('-------------------------------------------------------')
    

