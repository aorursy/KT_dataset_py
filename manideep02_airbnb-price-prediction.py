from IPython.display import Image

Image("/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png")
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

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.metrics import r2_score,mean_squared_error
data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head(2)
data.info()
len(data.columns),len(data)



# here we 16 columns and 48895 rows
data.dtypes
data.drop(["host_name","name"],axis=1,inplace=True)
data_num=data.select_dtypes(include=[int,float])   # since we have no float dtype column in dataframe

def detect_outlier(col):

    outlier=[]

    threshold=3

    mean=np.mean(col)

    stdeviation=np.std(col)

    

    for i in col:

        z_score=(i-mean)//stdeviation

        if z_score > threshold:

            outlier.append(i)

    

    return outlier
dict_for_numerical_columns={}

for i in data_num.columns:

    # intialising the dictionary

    dict_for_numerical_columns[i]=0   

for i in dict_for_numerical_columns:

    # sending each and every column in data_num

    l=detect_outlier(data_num[i])      

    dict_for_numerical_columns[i]=l



print(dict_for_numerical_columns)



# from this, we found that there are outliers in data
outlier=[]  # for storing the outlier columns

for i in dict_for_numerical_columns:

    if len(dict_for_numerical_columns[i])>1:

        print(f"the {i} has {len(dict_for_numerical_columns[i])} outliers")

        outlier.append(i)

print("*****outlier columns************")

print(*outlier)
fig=plt.figure(1,(20,5)) # 20 is length of x-axis in below figure and 5 is length of y axis

ax=plt.subplot(1,1,1)   #(1,1,1) denotes in 1x1 grid 1st plot

sns.boxplot(x="variable",y="value",data=pd.melt(data_num))

plt.show()
fig=plt.figure(11,(15,10))

for i,col in enumerate(data_num.columns):

    ax=plt.subplot(4,3,i+1)

    sns.boxplot(data_num[col],linewidth=1,palette="plasma")

    plt.tight_layout()

plt.show()
outlier # it is a list of all outliers
o1=data.iloc[list(data[outlier[0]][dict_for_numerical_columns["price"]].index)]

o1.head(2)
o2=data.iloc[list(data[outlier[1]][dict_for_numerical_columns['minimum_nights']].index)]

o2.head(2)
o3=data.iloc[list(data[outlier[2]][dict_for_numerical_columns[ 'number_of_reviews']].index)]



o3["number_of_reviews"].unique()
o4=data.iloc[list(data[outlier[3]][dict_for_numerical_columns['calculated_host_listings_count']].index)]

o4.head(2)
outliers_data=pd.concat([o1,o2,o3,o4])

# now we combined all outliers into a single dataframe

len(data),len(outliers_data),len(o1),len(o2),len(o3),len(o4)



# you will be shocked why we get 1701 when we all outliers' dataframes



# by concatenating,we do add duplicated rows too

# duplicate rows in outliers_data and original outliers_data rows

len(outliers_data[outliers_data.duplicated()]),len(outliers_data.index)
data.drop(outliers_data.index,axis=0,inplace=True)

# all rows are removed(duplicated too)
len(data),

# thus obtained rows are 48895-(1701-1382)=48576
data.isnull().sum()
data.drop("last_review",axis=1,inplace=True)
sns.countplot(x=data["reviews_per_month"],data=data)
data.reviews_per_month.isnull().sum()
data.dropna(how="any",inplace=True)

data.reviews_per_month.isnull().sum()

# hence we have no missing values
data.skew()

# the skew which are +ve are right skew

# the skew which are -ve are left skew
skew_columns=["id","host_id","price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]

fig=plt.figure(1,(15,9))

for i,col in enumerate(skew_columns):

    ax=plt.subplot(4,3,i+1)

    sns.distplot(data[col],kde=50)

    ax.set_xlabel(col)

    ax.set_title(f"{col} distribution")

    plt.tight_layout()

plt.show()

    
plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group)

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.price,palette="plasma")

plt.ioff()
data.head(3)
corr = data.corr(method='pearson')

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True,)

data.columns
# we are assuming that these columns are not useful to predict

data.drop(["latitude"],axis=1,inplace=True)

data.drop(["longitude"],axis=1,inplace=True)
sns.pairplot(data)
k=data

k.head()
data["room_type"].nunique(),data["neighbourhood_group"].nunique(),data["neighbourhood"].nunique()

# we dont consider for dummy values of neighbourhood_group and neighbourhood columns
data["room_type"].unique()
p=pd.get_dummies(data.room_type)

p.head()
data1=pd.concat([p,data],axis=1)

# after concatenating,we removed the original column

data1.drop("room_type",axis=1,inplace=True)

data1.head()
data1.head()
data1.drop(["neighbourhood_group","neighbourhood","host_id"],axis=1,inplace=True)

data1.head()
data1.head()
data1.drop("id",axis=1,inplace=True)

# let place the prize column at last

d=data1["price"]

data1.drop("price",axis=1,inplace=True)

data1["price"]=d
x=data1.iloc[:,0:8]

y=data1["price"]

model1=SelectKBest(score_func=chi2,k=7)

model1=model1.fit(x,y)

features1=model1.transform(x)
print(model1.scores_)

print("********************")



# we have to select the highest score column for best accuracy

# so the columns are shared_room,availability_365,calculated_host_listings_count,reviews_per_month,number_of_reviews,Private room
model1.scores_=pd.Series(model1.scores_,index=x.columns)

model1.scores_.nlargest(13).plot(kind="barh")

plt.show()
data1.columns
data1.drop(list(set(data1.columns)-set([ 'Shared room','Entire home/apt', 'Private room',

       'minimum_nights', 'number_of_reviews', 'reviews_per_month',

       'calculated_host_listings_count', 'availability_365'])),axis=1,inplace=True)
data1["price"]=y

data1.head()

# our new modified dataset
x=data1.iloc[:,0:5]

y=data1.iloc[:,5]
len(data.reviews_per_month)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x=scaler.fit_transform(x)

# y has only one column, we need to reshape to single array

y=np.array(y)

y=y.reshape(-1,1)

y=scaler.fit_transform(y)

from sklearn.ensemble import RandomForestRegressor

import math

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=101)

model=RandomForestRegressor(n_estimators=100,max_depth=15,random_state=0)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

math.sqrt(mean_squared_error(y_pred,y_test))