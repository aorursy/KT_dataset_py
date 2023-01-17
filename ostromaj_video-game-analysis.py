import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
vgdata = pd.read_csv('../input/videogames-sales-dataset/Video_Games_Sales_as_at_22_Dec_2016.csv')
#Dropping null values from columns where not a lot are missing
vgdata.dropna(subset=['Name'],inplace=True)
vgdata.dropna(subset=['Year_of_Release'],inplace=True)
vgdata.dropna(subset=['Publisher'],inplace=True)
vgdata.dropna(subset=['Genre'],inplace=True)

#Filling Null Values for larger missing data
vgdata['Developer'].fillna('None',inplace=True)
vgdata['Rating'].fillna('None',inplace=True)
print("Shape of DF")
print('-'*30)
print("ROWS: {}        COLUMNS: {}".format(vgdata.shape[0],vgdata.shape[1]))
vgdata.head()
vgdata.dtypes
print('Missing Values %')
print("-"*25)
print(round(vgdata.isnull().sum()/vgdata.shape[0]*100,2))
vgdata.describe()
#Sorting
vgdata.sort_values('NA_Sales',ascending=False).head(5)
#Groupby best game visualization
vgdata[['Platform','Name','NA_Sales','EU_Sales','JP_Sales','Other_Sales']]\
.loc[vgdata.groupby('Platform').NA_Sales.agg('idxmax')]\
.sort_values('NA_Sales',ascending=False)[0:10]\
.set_index('Name')\
.plot(kind='bar',stacked=True,figsize=(20,5))

plt.title("Most Sold games by platform")
plt.xlabel("Games")
plt.ylabel("Number Sold(Millions)")
#vgdata.groupby(['Platform','Name'], as_index=False)['Global_Sales'].max().sort_values("Global_Sales",ascending=False).head(10)
data=vgdata.Rating.value_counts()
data.plot(kind='pie')
data=vgdata.Developer.value_counts()[1:20]
data.plot(kind='bar',figsize=(20,5))
data=vgdata.Year_of_Release
data.plot.hist(by='Year_of_Release',bins=data.nunique(),figsize=(20,5))
data=vgdata.Platform.value_counts()[0:20]
data.plot(kind='bar',figsize=(20,5))
data=vgdata.Genre.value_counts()[0:20]
data.plot(kind='bar',figsize=(20,5))
data=vgdata.Publisher.value_counts()[0:20]
data.plot(kind='bar',figsize=(20,5))
import matplotlib.pyplot as plt
vgdata['NA_Sales'].plot(kind='kde',figsize=(20,5))
plt.xlim((-1,1))
sns.lmplot(x='Year_of_Release',y='NA_Sales',data=vgdata,size=8)
nicedata=vgdata[vgdata['NA_Sales']<0.5]
nicedata=nicedata[nicedata['NA_Sales']>-0.5]
fig=plt.gcf()
sns.boxplot(x='Genre',y='NA_Sales',data=nicedata)
fig.set_size_inches(20, 5)
fig=plt.gcf()
sns.boxplot(x='Platform',y='NA_Sales',data=nicedata)
fig.set_size_inches(20, 5)
fig=plt.gcf()
sns.boxplot(x='Rating',y='NA_Sales',data=nicedata)
fig.set_size_inches(20, 5)
#CLEAN DATASET
## Remove or Replace NULL VALUELS
## One-Hot Encoding/Label Encoding Categorical columns
#Drop Columns we will not use
vgdata.drop(["Critic_Score","Critic_Count","User_Score","User_Count","Global_Sales",'JP_Sales',"EU_Sales","Other_Sales"],
            axis=1,inplace=True)
vgdata.head()
#Checking cardinality

columns=['Platform','Year_of_Release','Genre','Publisher','Developer','Rating']
for i in columns:
    print(i)
    print("_"*20)
    print(vgdata[i].nunique())
    print('\n')
#OHE(Get_dummies) GENRE AND RATING
vgdata=pd.concat([vgdata,pd.get_dummies(vgdata.Rating)],axis=1)
vgdata=pd.concat([vgdata,pd.get_dummies(vgdata.Genre)],axis=1)
vgdata.drop(['Genre','Rating'],axis=1,inplace=True)
vgdata.head()
#LABEL ENCODE DEVELOPER PUBLISHER YEAR_OF RELEASE AND PLATFORM

Dev_LE=LabelEncoder()
vgdata.Developer=Dev_LE.fit_transform(vgdata.Developer)

Pub_LE=LabelEncoder()
vgdata.Publisher=Pub_LE.fit_transform(vgdata.Publisher)

YOR_LE=LabelEncoder()
vgdata.Year_of_Release=YOR_LE.fit_transform(vgdata.Year_of_Release)

Plat_LE=LabelEncoder()
vgdata.iloc[:,1]=Plat_LE.fit_transform(vgdata.iloc[:,1])

vgdata.rename(columns={vgdata.columns[1]:'System'}, inplace=True)

vgdata.head()
#Create Variables for ML
Y=vgdata['NA_Sales']
X=vgdata
X.drop(['NA_Sales'],axis=1,inplace=True)
X.drop('Name',axis=1,inplace=True)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
LR=LinearRegression()
#fit
LR.fit(X_train,y_train)
#predict
pred = LR.predict(X_test)
#scoring
print(mean_squared_error(y_test,pred))
