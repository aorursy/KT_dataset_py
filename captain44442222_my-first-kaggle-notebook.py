import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
warnings.simplefilter('ignore')
sns.set_style('whitegrid')
data=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
data.describe()
data[data['Rating']>5]
data=data.drop(10472)
data.isna().sum()
data=data.dropna()
data.info()
data['Reviews']=pd.to_numeric(data['Reviews'])
def size(x):
    if x[-1]=='k':
        b=round(float(x[0:-1])*1000)
    elif x[-1]=='M':
        b=round(float(x[0:-1])*1000000)
    elif x=='Varies with device': 
        b=np.nan 
    return(b)
data['Size']=data['Size'].apply(size)
data['Size']=data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'))
data['Size']=pd.to_numeric(data['Size'])
data['Installs']=data['Installs'].astype(str).str.replace('+','')
data['Installs']=data['Installs'].astype(str).str.replace(',','')
data['Installs']=pd.to_numeric(data['Installs'])
data['Price']=data['Price'].apply(lambda x: float(str.replace(x,'$','')))
data['Android_ver2']=data['Android Ver'].apply(lambda x : x.split()[0])
data.info()
data['Rating'].hist(figsize=(12,8),bins=30)
plt.title('Distribution of Ratings',size=18)
plt.figure(figsize=(15,8))
for i in data['Type'].unique():
    sns.distplot(data[data['Type']==i]['Rating'],label=i)
    plt.legend()
plt.title('Distribution of rating in paid vs free APPs',size=18)
### as you can see most of applications still support Android 4.1 
plt.figure(figsize=(15,8))
sns.countplot(y=data['Android Ver'].apply(lambda x : x.split()[0]).sort_values())
plt.title('Lowest Android Version Supported By Apps',size=18)
plt.figure(figsize=(14,8))
sns.heatmap(data.groupby([pd.cut(data['Size'],bins=10),'Type'])['Rating'].mean().unstack())
plt.title('Heatmap of mean of rating for range of sizes & Type ',size=18)
plt.figure(figsize=(14,8))
sns.heatmap(data.groupby(['Category','Type'])['Rating'].mean().unstack())
plt.title('Heatmap of mean of rating based on Category & Type ',size=18)
plt.figure(figsize=(14,8))
sns.heatmap(data.groupby(['Genres','Type'])['Rating'].mean().unstack())
plt.title('Heatmap of mean of rating based on Genres & Type ',size=18)
data['Category'].value_counts().plot.pie(figsize=(15,15),autopct='%1.1f%%')
plt.title('Category Share in APPS',size='18')
plt.ylabel('Category',size=16)
data3=data.copy()
data3=data3[data3['Android_ver2']!='Varies']
data3['Android_ver2']=data3['Android_ver2'].apply(lambda x: int(x[0]))
plt.figure(figsize=(14,8))
data3.groupby('Category')['Android_ver2'].mean().plot(kind='bar')
### comparing distribution diffrent range of size in Rating
plt.figure(figsize=(15,8))
for i in range(0,int(data['Size'].max()),int(data['Size'].max()/10)):
    A=data[data['Size']<i][data['Size']>i-200000000]
    sns.distplot(A['Rating'],label=(i,'to',i+int(data['Size'].max()/10)),hist=False)
### distribution of rating in paid & free
plt.figure(figsize=(20,8))
for i in data['Category'].unique()[10:20]:
    sns.distplot(data[data['Category']==i]['Rating'],label=i,hist=False)
    plt.legend()
plt.figure(figsize=(20,8))
for i in data['Category'].unique()[20:]:
    sns.distplot(data[data['Category']==i]['Rating'],label=i,hist=False)
    plt.legend()
### lets check distribution of installs
plt.figure(figsize=(15,8))
ax=sns.countplot(data['Installs'])
ax.tick_params(labelsize=10)
plt.xticks(rotation=40)
plt.title('Distribution of Number of Installs',size=18)
pd.DataFrame(data.groupby('Category').mean()['Rating'])
sns.heatmap(data.corr(),annot=True)
sns.pairplot(data,kind='reg')
sns.lmplot('Rating','Reviews',data)
plt.figure(figsize=(15,8))
ax=sns.lvplot(x='Category',y='Rating',data=data)
ax.tick_params(labelsize=10)
plt.xticks(rotation=90)
plt.title('Comparing distribution of Rating in diffrenet categories',size=18)
plt.figure(figsize=(15,8))
ax=sns.boxenplot(x='Category',y='Size',data=data)
ax.tick_params(labelsize=10)
plt.xticks(rotation=90)
plt.title('Comparing distribution of Sizes in diffrenet categories',size=18)
plt.figure(figsize=(18,10))
ax=sns.boxplot(y='Rating',x='Content Rating',data=data)
#ax=sns.swarmplot(y='Rating',x='Content Rating',data=data)
ax.tick_params(labelsize=10)
plt.xticks(rotation=90)
plt.title('Comparing Rating based on Content Rating',size=18)
plt.figure(figsize=(18,36))
ax=sns.boxplot(x='Rating',y='Category',data=data,hue='Type')
ax.tick_params(labelsize=10)
plt.xticks(rotation=90)
plt.title('Boxplot of Rating vs Category based on Type of The APP',size=18)
data.describe()
### mean of data 4.2 
data.groupby('Installs').mean()['Rating'].plot(kind='bar',figsize=(12,8))
plt.title('Mean of Rating VS Installs',size=18)
data.groupby('Type')['Rating'].agg(['count','mean'])

data[data['Installs']>500000].groupby('Type')['Rating'].agg(['count','mean'])
data[data['Type']=='Free']['Rating'].hist(figsize=(12,8),bins=30,alpha=0.5)
data[data['Type']=='Paid']['Rating'].hist(figsize=(12,8),bins=30,alpha=0.5)
plt.title('Distribution of rating in paid vs free APPs',size=18)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
data2=data.drop(columns=['App','Current Ver','Android Ver','Genres','Last Updated','Type','Android_ver2'])
data2=pd.get_dummies(data2)
data2.head()
X=data2.drop(columns='Rating')
y=data2['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)
model = DecisionTreeRegressor(random_state=0)
model.fit(np.array(X_train), np.array(y_train).reshape(-1,1))
predictions=model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions),'\n')

model = RandomForestRegressor(max_depth=2, random_state=0)
model.fit(np.array(X_train), np.array(y_train).reshape(-1,1))
predictions=model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions),'\n')
