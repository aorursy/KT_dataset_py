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
dataset = pd.read_csv('../input/bigmart-sales-data/Train.csv')
dataset.describe()
dataset.info()
dataset.shape
dataset.isnull().sum()
dataset.head()
print('The unique items in the dataset is',dataset.Item_Identifier.nunique())

print('Total Items in the dataset is',dataset.Item_Identifier.count())
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
sns.distplot(dataset.Item_Outlet_Sales,bins = 50)

#The sales is skewed to the left 
#squareroot transformation makes the target variable normal

sns.distplot(np.sqrt(dataset.Item_Outlet_Sales),bins = 50)
corr = dataset.corr()

corr
corr.Item_Outlet_Sales.sort_values()

#Item_MRP is most correlated and Item_visiblity is negatively correlated implies lesser visiblity more sales
fig,ax = plt.subplots(figsize = (10,8))

sns.heatmap(corr,vmax=.8)
categorical_features = dataset.select_dtypes(include=[object]).columns
categorical_features
dataset.Item_Fat_Content.value_counts().plot(kind='bar')

#LF and reg has to be replaced with low fat and regular
sns.countplot(dataset.Item_Type)

plt.xticks(rotation = 90)
sns.countplot(dataset.Outlet_Size)

#In the dataset there are more medium and small stores 
sns.countplot(dataset.Outlet_Location_Type)

#More stores in the tier 3
sns.countplot(dataset.Outlet_Type)

plt.xticks(rotation=90)

#supermarket type 1 is most and rest are very lesser in number compared to it
sns.pairplot(dataset)

#Distribution of the target variable with the numerical features
plt.scatter(dataset.Item_Visibility, dataset.Item_Outlet_Sales, alpha = 0.3)

#Item visiblity is negatively correlated viewing the distribution in the below plot
dataset.pivot_table(index = 'Item_Type',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar',figsize = (10,8)),
dataset.pivot_table(index = 'Outlet_Establishment_Year',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')

#Year 1998 has the lowest sales otherwise rest of the values are same in the dataset
dataset.groupby('Item_Fat_Content').median()['Item_Outlet_Sales'].plot(kind='bar')
fig,axes = plt.subplots(figsize=(10,8))

sns.barplot(x=dataset.Outlet_Identifier,y = dataset.Item_Outlet_Sales,hue =dataset['Outlet_Type'])

plt.xticks(rotation = 90)
dataset.pivot_table(index = 'Outlet_Identifier',values = 'Item_Outlet_Sales',columns='Outlet_Type',aggfunc = np.median).plot(kind='bar',figsize= (12,8))

plt.xticks(rotation = 0)
dataset.pivot_table(index = 'Outlet_Size',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')

#Medium size has more sales based on the below visualisation
dataset.pivot_table(index = 'Outlet_Type',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')

#Supermarket type 3 has highest impact on the sales
dataset.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median).plot(kind='bar')
#Import mode function:

from scipy.stats import mode

#Determing the mode for each

outlet_size_mode = dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())

outlet_size_mode
def impute_size_mode(cols):

    Size = cols[0]

    Type = cols[1]

    if pd.isnull(Size):

        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]

    else:

        return Size

print ('Orignal #missing: %d'%sum(dataset['Outlet_Size'].isnull()))

dataset['Outlet_Size'] = dataset[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)

print ('Final #missing: %d'%sum(dataset['Outlet_Size'].isnull()))
#Determine the average weight per item:

#item_avg_weight = dataset.pivot_table(values='Item_Weight', index='Item_Identifier')



#Get a boolean variable specifying missing Item_Weight values

#miss_bool = dataset['Item_Weight'].isnull() 

#miss_bool

#Impute data and check #missing values before and after imputation to confirm

#dataset.loc[miss_bool,'Item_Weight'] = dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
dataset['Item_Weight'][dataset['Item_Weight'].isnull()]=dataset['Item_Weight'].mean()
dataset.info()
dataset.loc[dataset['Item_Visibility']==0.000000,'Item_Visibility']=np.mean(dataset['Item_Visibility'])
dataset['Outlet_Years'] = 2013 - dataset['Outlet_Establishment_Year']

dataset['Outlet_Years'].describe()
dataset['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace= True)
dataset.drop(['Item_Identifier','Outlet_Establishment_Year','Outlet_Identifier'],axis=1,inplace = True)
X = dataset.drop(['Item_Outlet_Sales'],axis=1)

y= dataset['Item_Outlet_Sales']
X.info()
X = pd.get_dummies(X)
from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
Regressor.fit(X_train,y_train)
y_pred = Regressor.predict(X_test)
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,y_pred)
np.sqrt(mse)
from sklearn.model_selection import cross_val_score

score=cross_val_score(Regressor,X_train,y_train,cv=10,scoring='neg_mean_squared_error')

np.mean(np.sqrt(-score))
from sklearn.linear_model import Ridge

r=Ridge(alpha=0.05,solver='cholesky')

r.fit(X_train,y_train)

predict_r=r.predict(X_test)

mse=mean_squared_error(y_test,predict_r)

r_score=np.sqrt(mse)

r_score
r=Ridge(alpha=0.05,solver='cholesky')

score=cross_val_score(Regressor,X_train,y_train,cv=10,scoring='neg_mean_squared_error')

r_score_cross=np.sqrt(-score)

np.mean(r_score_cross),np.std(r_score_cross)