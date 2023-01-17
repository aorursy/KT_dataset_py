# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle//input/bigmart-sales-data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data = pd.read_csv('/kaggle//input/bigmart-sales-data/Test.csv')

train_data = pd.read_csv('/kaggle//input/bigmart-sales-data/Train.csv')
print(train_data.shape)

print(test_data.shape)



train_data.head()
train_data.info()
train_data.isnull().sum()
test_data.columns
test_data.isnull().sum()
train_data.select_dtypes(include='object').nunique()
train_data.head()
import seaborn as sns

sns.barplot(x='Item_Fat_Content',y='Item_Outlet_Sales',data=train_data)
sns.set(rc={'figure.figsize':(20,8)})

chart = sns.barplot(x='Item_Type',y='Item_Weight',data=train_data)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
sns.set(rc={'figure.figsize':(10,8)})

sns.barplot(x='Outlet_Type',y='Item_Outlet_Sales',data=train_data)
train_data['Item_Weight'].describe()
train_data['source'] = 'train'

test_data['source'] = 'test'

test_data['Item_Outlet_Sales'] = 0

data = pd.concat([train_data, test_data], sort = False)
#Item weight to be filled with mean value

mean = data['Item_Weight'].mean()

data['Item_Weight'] = data['Item_Weight'].fillna(value=mean)
data.isnull().sum()
sns.distplot(data['Item_Outlet_Sales'])
sns.set(rc={'figure.figsize':(10,8)})

sns.barplot(x='Outlet_Size',y='Item_Outlet_Sales',data=train_data)
sns.countplot('Outlet_Size',data=data)
from scipy.stats import mode



#Determing the mode for each

outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]))

print ('Mode for each Outlet_Type:')

print (outlet_size_mode)



#Get a boolean variable specifying missing Item_Weight values

missing_values = data['Outlet_Size'].isnull() 



#Impute data and check #missing values before and after imputation to confirm

print ('\nOrignal #missing: %d'% sum(missing_values))

data.loc[missing_values,'Outlet_Size'] = data.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

print (sum(data['Outlet_Size'].isnull()))
data.head()
data['Item_Fat_Content'].unique()
#Lets merge contents to Low fat and Regular

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})

#Item visibilty must be some value as 0 visibilty didnt make any sense , as with 0 visibilty the product  outlet sale should be zero, but is isn't.

#Hence let change the zero value.

sns.scatterplot('Item_Visibility','Item_Outlet_Sales',data=data)

#We are gonna fill 0 values with mean.

mean = data['Item_Visibility'].mean()

data=data.replace({'Item_Visibility': {0.0: mean}})
data['Item_Type'].unique()
data.head()
dummies = pd.get_dummies(data[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']])

dummies
data_processed = pd.concat([data,dummies],axis=1)
data_processed.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#New variable for outlet

data_processed['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
data_processed.drop(['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type'],inplace=True,axis=1)
data_processed.head()
data_processed['Outlet_Year'] = 2009 - data_processed['Outlet_Establishment_Year']
data_processed.drop(['Outlet_Establishment_Year'],axis=1,inplace=True)
train = data_processed.loc[data['source']=="train"]

test = data_processed.loc[data['source']=="test"]



#Drop unnecessary columns:

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)

train.head()
print(train.shape)

print(test.shape)