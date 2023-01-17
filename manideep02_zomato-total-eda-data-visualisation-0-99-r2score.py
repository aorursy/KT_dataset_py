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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split 

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score
data=pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")

data.head(3)
print(len([data.menu_item[i]=='[]' for i in range(len(data))]),len(data))



# since all rows are empty,we are going to remove the columns

data.drop("menu_item",axis=1,inplace=True)



data.drop(["url","address","phone"],axis=1,inplace=True)
data.isnull().sum()
data.info()
# we are removing the null values' rows in any of the below columns

data.dropna(subset=["location","cuisines","rest_type"],how="any",inplace=True)
data.rate=data.rate.astype(str)  



data.rate=data.rate.loc[data.rate!="NEW"]  # we have to select the integer string values only



data.rate=data.rate.loc[data.rate!="-"]



data.rate=data.rate.str.replace("/5","")  # removing "/5" in integer string



data.rate=data.rate.apply(lambda x:float(x)) # converting to float



myimputer=SimpleImputer(strategy="mean") # replacing null values with mean



data.rate=pd.DataFrame(myimputer.fit_transform(np.array(data.rate).reshape(-1,1)))
# "dish_liked" column

# Now we are going to use simple imputer to replace the missing values



myimputer=SimpleImputer(strategy="most_frequent")



data.dish_liked=pd.DataFrame(myimputer.fit_transform(np.array(data.dish_liked).reshape(-1,1)))



data.head(2)
# for approx_cost column



# we have convert it to float type





data['approx_cost(for two people)']=data['approx_cost(for two people)'].str.replace(",","") # removing , in integers



data['approx_cost(for two people)']=data['approx_cost(for two people)'].apply(lambda x:float(x)) # converting to float



myimputer=SimpleImputer(strategy="mean")  # replacing the missing value with mean value



data['approx_cost(for two people)']=pd.DataFrame(myimputer.fit_transform(np.array(data['approx_cost(for two people)']).reshape(-1,1)))
data.dropna(how="any",subset=list(data.columns),inplace=True)
data[data.duplicated()]

# since we have single duplicated row,we neglect it or we can remove the row
data.drop_duplicates(keep="first").head(2)  # we keep only the first occured row
data2=data.groupby("name").name.agg(["count"]).reset_index()



# renaming columns as count column which is obtained is function name.so we cant use it



data2.rename(columns={"name":"name","count":"repetition"},inplace=True)



print(data2.head())



data2.name[data2.repetition.max()] # the restaurant having more number of branches
maximum=data.votes.max()



# renaming columns as count column which is obtained is function name.so we cant use it



data.name[data.votes==maximum]



# The Byg Brewski Brewing Company restaurant has highest votes with 3 different branches
maximum=data.rate.max()



print(maximum)



# renaming columns as count column which is obtained is function name.so we cant use it



len(data.name[data.rate==maximum])



# There 55 highest rated restaurants
plt.figure(1,(20,10))

sns.countplot(data.book_table,data=data,hue="listed_in(type)")

plt.show()



# most of restaurants dont allow booking table 
plt.figure(1,(20,10))

sns.countplot(data.online_order,data=data,hue="listed_in(type)")

plt.show()
plt.figure(1,(30,20))

ax=sns.countplot(data["listed_in(city)"],data=data)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,ha="right")

plt.show()
#Restaurant Type

ax=sns.countplot(data['rest_type'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()   # it is used to get the current figure.if we dont get,figure() function creates the one

fig.set_size_inches(15,15)   # sets the inches of rest_type

plt.title('Restuarant Type')
plt.figure(1,(10,10))

data2=data.name.value_counts()[:15]

ax=sns.barplot(x=data2.index,y=data2,palette="bright") # different types of palettes--bright,muted,dark,colorblind

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.ylabel("rate")

plt.show()
plt.figure(1,(15,7))

ax=sns.countplot(data.rate,data=data,hue="listed_in(type)")

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.show()
#Encode the input Variables

def Encode(dat):

    for column in dat.columns[~dat.columns.isin(['rate', 'cost', 'votes'])]:

        dat[column] = dat[column].factorize()[0]   # it used for the numeric representation of an array to identify the distinct values

    return dat
finaldata = Encode(data.copy())

finaldata.head()
corr=finaldata.corr(method="pearson")

plt.figure(1,(15,8))

sns.heatmap(corr,annot=True)

plt.show()


x = finaldata.iloc[:,[2,3,8,9,11]]

y = finaldata['approx_cost(for two people)']

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=101)

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(min_samples_leaf=0.01,random_state=101) 

# when min_smaples_leaf is int,then it is number required to considered as leaf

# if it is float, required number of leaf =len(data)*0.00007  her we get 3

model.fit(x_train,y_train)

pred=model.predict(x_test)

r2_score(pred,y_test)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=500,random_state=101,min_samples_leaf=.0001)

rf.fit(x_train,y_train)   # n_estimators--The number of trees in forest

y_predict=rf.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_predict)