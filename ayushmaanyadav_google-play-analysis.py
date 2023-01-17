import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import datetime as dt

import scipy.stats as sc

import warnings
dataset = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv",encoding='utf-8')
dataset.head()
dataset.info()
dataset.describe()
#Null values in the respective Column 

temp = dataset.isnull().sum()

temp
# Calculating Missing Values Percentage

temp = (temp / dataset.shape[0])*100

temp
# removing the null values from the dataset to calculate mean,median and mode

t_dataset = dataset[pd.notnull(dataset['Rating'])]

print(t_dataset.info())
# mean,median and mode of rating and ploting the graph to find Skewness of the graph of the not null data

mean_t = np.mean(t_dataset["Rating"])

median_t = np.median(t_dataset["Rating"])

mode_t = sc.mode(t_dataset["Rating"])

print("Printing the values of Mean, Median, Mode respectively :",mean_t,median_t,mode_t[0][0])

dataset["Rating"].plot(kind='hist')

plt.grid()

plt.xlim(0,7)

plt.show()

# from this we can see that the data is right skewed so we can replace none value by meadian of dataset 
# Replaced the null values of rating with median of rating

dataset['Rating'].fillna(median_t,inplace = True)

dataset.info()
# Dropping the rest null values as they have negligible effect on the dataset so we can drop it.

dataset.dropna(inplace= True)

dataset.info()
#  checking duplicate value percentage in the data(true represent duplicate values) 

temp = dataset.duplicated().value_counts()

print(temp)

temp = temp/dataset.shape[0] * 100

print(temp)

# There are 483 duplicate values in the dataset which are 4.4 percent of the overall dataset 
# removing the dupilicate Vaues of the table

dataset.drop_duplicates(inplace = True)

dataset.info()
# Now the data has been cleaned 

dataset.describe()
# Converting last updated to days spend from last update

dataset['Last Updated'] = pd.to_datetime(dataset['Last Updated'])

dataset['Days Before Updated'] = (dataset['Last Updated'].max() - dataset['Last Updated'])
# Removing the , and + from the install

dataset['Installs'] = dataset['Installs'].str.replace(',','').str.replace('+','').astype('int')
# Converting Review form string to int type

dataset['Reviews'] = dataset['Reviews'].astype(int)
# Price to int

dataset['Price'] = dataset['Price'].str.replace('$','').astype('float')
# Conveting size to a common scale of measurement

dataset['Size'] = dataset['Size'].str.replace('Varies with device',"0").str.replace('M','e+6').str.replace('k','e+3').astype('float')
dataset.describe()
plt.figure(figsize = (10,10))

dataset["Category"].value_counts().plot(kind='pie')

plt.show()

plt.figure(figsize = (10,5))

dataset["Category"].value_counts().plot(kind='bar')

plt.ylabel("No of Apps")

plt.xlabel("Category of Apps")

plt.grid()

plt.show()
plt.figure(figsize = (5,5))

dataset["Content Rating"].value_counts().plot(kind='pie',autopct = "%f%%")

plt.legend()

plt.show()

dataset["Content Rating"].value_counts().plot(kind='bar')

plt.ylabel("No of Apps")

plt.xlabel("Types of Content Rating")

plt.grid()

plt.show()
plt.figure(figsize = (40,10))

dataset["Genres"].value_counts().plot(kind='bar')

plt.ylabel('No of Apps')

plt.xlabel('Type of Genres')

plt.show()
plt.figure(figsize = (5,5))

dataset["Type"].value_counts().plot(kind='pie',autopct = "%i%%")

plt.legend()

plt.show()

dataset["Type"].value_counts().plot(kind='bar')

plt.grid()

plt.xlabel("Type of Apps Paid or Free")

plt.ylabel("No of Apps")

plt.show()
dataset[dataset['Size'] == dataset['Size'].max()]
dataset[dataset['Installs'] == dataset['Installs'].max()]
dataset[dataset['Days Before Updated'] == dataset['Days Before Updated'].max()]
dataset[dataset['Reviews'] == dataset['Reviews'].max()]
sorte = dataset.sort_values(['Reviews'],ascending = 0 )[:20]

ax = sns.barplot(x = 'Reviews' , y = 'App' , data = sorte )

ax.set_xlabel('Reviews')

ax.set_title("Most Reviewed Apps in Play Store", size = 20)
sort_data = dataset[dataset['Category'] == 'FAMILY'].sort_values(['Installs'],ascending= 0)[:15]

ax = sns.barplot(x = 'Installs' , y = 'App' , data = sort_data )

ax.set_xlabel('Installs')

ax.set_title("Most Installed Apps in Play Store", size = 20)
sort_data = dataset[dataset['Category'] == 'GAME'].sort_values(['Installs'],ascending= 0)[:30]

ax = sns.barplot(x = 'Installs' , y = 'App' , data = sort_data )

ax.set_xlabel('Installs')

ax.set_title("Most Installed Apps in Play Store", size = 20)
sort_data = dataset[dataset['Category'] == 'TOOLS'].sort_values(['Installs'],ascending= 0)[:20]

ax = sns.barplot(x = 'Installs' , y = 'App' , data = sort_data )

ax.set_xlabel('Installs')

ax.set_title("Most Installed Apps in Play Store", size = 20)
sort_data = dataset[dataset['Category'] == 'BUSINESS'].sort_values(['Installs'],ascending= 0)[:15]

ax = sns.barplot(x = 'Installs' , y = 'App' , data = sort_data )

ax.set_xlabel('Installs')

ax.set_title("Most Installed Apps in Play Store", size = 20)
plt.figure(figsize = (10,10))

dataset['Android Ver'].value_counts().plot(kind= 'pie')

plt.show()

plt.figure(figsize = (10,5))

dataset['Android Ver'].value_counts().plot(kind= 'bar')

plt.grid()

plt.show()
dataForModeling = dataset[['Price','Reviews','Size','Installs','Days Before Updated','Category','Type','Content Rating','Genres']]
data_X = pd.get_dummies(dataForModeling, columns=['Category',"Content Rating","Type","Genres"])

scaler=MinMaxScaler()

scaler.fit(data_X[['Days Before Updated']])

data_X[['Days Before Updated']]=scaler.transform(data_X[['Days Before Updated']])
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(data_X,dataset['Rating'],random_state=0)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
x_train.shape,x_test.shape
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_train, y_train)

reg.score(x_train, y_train)
reg.score(x_test, y_test)
lr_df= pd.DataFrame(data = {"Predicted": reg.predict(x_test) , "Actual": y_test})
plt.plot(lr_df["Predicted"][:30], "*")

plt.plot(lr_df['Actual'][:30], "^")

plt.show()
from sklearn.svm import SVR
regr = SVR(C=2.0, epsilon=0.3)

regr.fit(x_train, y_train)

print(regr.score(x_train, y_train))

print(regr.score(x_test, y_test))
svr_df= pd.DataFrame(data = {"Predicted": regr.predict(x_test) , "Actual": y_test})
plt.plot(svr_df["Predicted"][:30],"*")

plt.plot(svr_df['Actual'][:30], "^")

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state = 0,max_depth = 8)

rfr.fit(x_train, y_train)

print(rfr.score(x_train, y_train))

rfr.score(x_test, y_test)
rfr_df= pd.DataFrame(data = {"Predicted": rfr.predict(x_test) , "Actual": y_test})
plt.plot(svr_df["Predicted"][:30],"*")

plt.plot(lr_df["Predicted"][:30], "*")

plt.plot(rfr_df["Predicted"][:30],"*")

plt.plot(rfr_df['Actual'][:30], "^")

plt.show()