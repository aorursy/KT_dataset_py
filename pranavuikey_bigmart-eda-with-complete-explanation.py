import pandas as pd 

import numpy as np                     # For mathematical calculations 

import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 

import warnings   # To ignore any warnings 

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/bigmart-sales-data/Train.csv')

test = pd.read_csv('../input/bigmart-sales-data/Test.csv')
train.shape,test.shape
train_original=train.copy() 

test_original=test.copy()
train.info(),test.info()
train['source'] = 'train'

# test['source'] = 'test'

test['Item_Outlet_Sales'] = 0

data = pd.concat([train, test], sort = False)

print(train.shape, test.shape, data.shape)
data['Item_Outlet_Sales'].describe()
sns.distplot(data['Item_Outlet_Sales'])
print('Skewness: %f' % data['Item_Outlet_Sales'].skew())

print('Kurtsis: %f' %data['Item_Outlet_Sales'].kurt())
categorial_features = data.select_dtypes(include=[np.object])

categorial_features.head(2)
numerical_features = data.select_dtypes(include=[np.number])

numerical_features.head(2)
data['Outlet_Establishment_Year'].value_counts()
train['Item_Outlet_Sales'].hist(bins = 100);
train['Item_Weight'].hist(bins = 100);
train['Item_Visibility'].hist(bins = 100)
train['Item_MRP'].hist(bins = 100)
import seaborn as sns

sns.catplot(x="Item_Fat_Content", kind="count", data=train);
train['Item_Fat_Content'].replace({'reg':'Regular','low fat':'Low Fat','LF':'Low Fat'},inplace = True)
sns.catplot('Item_Fat_Content',kind = 'count',data = train)
sns.catplot('Item_Type',kind = 'count',data = train,aspect =4)
sns.catplot('Outlet_Identifier',kind = 'count',data = train,aspect = 2)
sns.catplot('Outlet_Size',kind = 'count',data = train,aspect = 2)
sns.catplot('Outlet_Establishment_Year',kind = 'count',data = train,aspect =4)
sns.catplot('Outlet_Type',kind = 'count',data = train,aspect =4)
sns.scatterplot(x = 'Item_Weight',y = 'Item_Outlet_Sales',data = train,alpha = 0.3);
sns.scatterplot(x = 'Item_Outlet_Sales',y = 'Item_Visibility',data = train,alpha = 0.3)
sns.scatterplot(x = 'Item_MRP',y = 'Item_Outlet_Sales',data = train,alpha = 0.3)
sns.catplot(x = 'Item_Type',y = 'Item_Outlet_Sales',kind = 'violin',data = train,aspect=4)
train.info()
sns.violinplot(x = 'Item_Fat_Content',y = 'Item_Outlet_Sales',data = train)
sns.catplot('Outlet_Identifier','Item_Outlet_Sales',kind = 'violin',data = train,aspect = 4)
sns.violinplot('Outlet_Size','Item_Outlet_Sales',data = train)
sns.violinplot('Outlet_Location_Type','Item_Outlet_Sales',data = train)
sns.violinplot('Outlet_Type','Item_Outlet_Sales',data = train)
train.isna().sum()
from sklearn.preprocessing import LabelEncoder

l_enc  = LabelEncoder()

a = l_enc.fit_transform(train['Item_Identifier'])
a
train['Item_Weight'].fillna(a.mean(),inplace = True)
train.Item_Weight.isna().sum()
train['Outlet_Size'].fillna('Small',inplace  = True)
train['Outlet_Size'].isna().sum()
train['Item_Visibility'].plot(kind = 'hist',bins = 100)
train.shape
a= train[train['Item_Visibility']!=0]['Item_Visibility'].mean()
train['Item_Visibility'] = train['Item_Visibility'].replace(0.00,a)
train['Item_Visibility'].plot(kind = 'hist',bins = 100)
perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]

non_perishable = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks"]
item_list =[] 

for i in train['Item_Type']:

    if i in perishable:

        item_list.append('perishable')

    elif (i in non_perishable):

        item_list.append('non_perishable')

    else:

        item_list.append('not_sure')

        

train['Item_Type_new'] = item_list
train['Item_Category'] =train['Item_Identifier'].replace({'^DR[A-Z]*[0-9]*':'DR','^FD[A-Z]*[0-9]*':'FD','^NC[A-Z]*[0-9]*':'NC'},regex = True)
Food=pd.crosstab(train['Item_Type'],train['Item_Category'])

Food
train['Item_Fat_Content'][(train['Item_Category']=='NC')]='Non Edible'
train['Item_Fat_Content'].unique()
train['Outlet_Years'] = 2019-train['Outlet_Establishment_Year']
train['Price_Per_Unit_Weight'] = train['Item_MRP']/train['Item_Weight']
def clusters(x):

    if x<69:

        return '1st'

    elif x in range(69,136):

        return '2nd'

    elif x in range(136,203):

        return '3rd'

    else:

        return '4th'

train['Item_MRP_Clusters'] = train['Item_MRP'].astype('int').apply(clusters)

train.head()
train['Item_MRP_Clusters'].unique()
from sklearn.preprocessing import LabelEncoder
# a = ['Outlet_Size','Outlet_Location_Type']

le = LabelEncoder()

train['Outlet_Size']= le.fit_transform(train['Outlet_Size'])

train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])

train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])

train['Item_MRP_Clusters'] = le.fit_transform(train['Item_MRP_Clusters'])
train.info()
#train['Outlet_Identifier'].unique(),train['Item_Identifier'].unique(),train['Item_Type'].unique()

a = pd.get_dummies(train[['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Item_Type_new','Item_Category']])

train = train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Item_Type_new','Item_Category','source'],axis = 1 )

train = pd.concat([train,a],axis = 1)
train['Price_Per_Unit_Weight'] = np.log(train['Price_Per_Unit_Weight'])

train['Item_Visibility'] = np.log(train['Item_Visibility'])
corr = train.corr()

corr
from sklearn.linear_model import LinearRegression
X = train.drop('Item_Outlet_Sales',axis = 1)

y = train['Item_Outlet_Sales']
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state = 8)
print(X_train.shape,X_valid.shape,y_train.shape,y_valid.shape)
X_train.head()
model = LinearRegression(normalize=True,fit_intercept= True)
model.fit(X,y)
y_pred = model.predict(X_valid)
model.score(X_train,y_train),model.score(X_valid,y_valid)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mean_squared_error(y_valid, y_pred),mean_absolute_error(y_valid, y_pred)
from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train,y_train)

predictions = my_model.predict(X_valid)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(y_valid,predictions)))
from sklearn.linear_model import Lasso
ls = Lasso(alpha = 0.01)

ls.fit(X_train,y_train)

predictions = ls.predict(X_valid)
mean_absolute_error(y_valid,predictions)