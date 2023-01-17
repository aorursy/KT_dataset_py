import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

pd.pandas.set_option('display.max_columns',None)
data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.head()
data.drop(labels=['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
data.head()
data.drop('Genres',axis=1,inplace=True)
data['Category'].value_counts()
data.head()
data['Category'].unique()
data.isnull().sum()
data['Rating'] = data['Rating'].fillna(data['Rating'].median())

data['Content Rating']=data['Content Rating'].fillna(data['Content Rating'].mode())

data.head()
data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()
sns.catplot(x='Category',y='Rating',data=data.sort_values('Rating',ascending=False),kind='boxen',height=6,aspect=3)
data.head()
data.dtypes
data['Reviews']=data['Reviews'].astype(int)
data.dtypes
data.tail()
#scaling and cleaning size of installation

def change_size(size):

    if 'M' in size:

        x = size[:-1]

        x = float(x)*1000000

        return(x)

    elif 'k' == size[-1:]:

        x = size[:-1]

        x = float(x)*1000

        return(x)

    else:

        return None



data["Size"] = data["Size"].map(change_size)



#filling Size which had NA

data.Size.fillna(method = 'ffill', inplace = True)
data.head()
data['Type'].value_counts()    
#Cleaning no of installs classification

data['Installs'] = [int(i[:-1].replace(',','')) for i in data['Installs']]
data.head()
Category=data['Category']

Category=pd.get_dummies(Category,drop_first=True)
Category.head()
Category.tail()
data['Content Rating'].value_counts()
Content =data['Content Rating']

Content=pd.get_dummies(Content,drop_first=True)

Content.head()
#Cleaning prices

def price_clean(price):

    if price == '0':

        return 0

    else:

        price = price[1:]

        price = float(price)

        return price



data['Price'] = data['Price'].map(price_clean).astype(float)
Type=data['Type']

Type=pd.get_dummies(Type,drop_first=True)

Type.head()
data.head()
train_data=pd.concat([data,Category,Type,Content],axis=1)
train_data.head()
train_data.drop(labels=['Category','Type','Content Rating'],axis=1,inplace=True)
train_data.head()
train_data.columns
X=train_data.loc[:,['Reviews', 'Size', 'Installs', 'Price', 'AUTO_AND_VEHICLES',

       'BEAUTY', 'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',

       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY', 'FINANCE',

       'FOOD_AND_DRINK', 'GAME', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',

       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'MEDICAL',

       'NEWS_AND_MAGAZINES', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY',

       'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS',

       'TRAVEL_AND_LOCAL', 'VIDEO_PLAYERS', 'WEATHER', 'Paid', 'Everyone',

       'Everyone 10+', 'Mature 17+', 'Teen', 'Unrated']]
y=train_data.iloc[:,0]
y.head()
X.head()
from sklearn.ensemble import ExtraTreesRegressor

reg=ExtraTreesRegressor()

reg.fit(X,y)
print(reg.feature_importances_)
plt.figure(figsize=(12,8))

feat_importances=pd.Series(reg.feature_importances_,index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor

rg=RandomForestRegressor()

rg.fit(X_train,y_train)
y_pred=rg.predict(X_test)
y_pred
rg.score(X_train,y_train)
rg.score(X_test,y_test)
sns.distplot(y_test-y_pred)
from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))

print('MSE:',metrics.mean_squared_error(y_test,y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
metrics.r2_score(y_test,y_pred)
    