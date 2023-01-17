import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/googleplaystore.csv")
data.info()
data.isnull().sum()
data.isnull().sum()*100/len(data)
data.isnull().sum(axis = 1)*100/len(data)
data.info()
data.dropna(inplace=True)
data["Reviews"]=data["Reviews"].astype(str).str.replace(",","")

data["Reviews"]=data["Reviews"].astype(str).str.replace("M", "")

data["Reviews"] = data["Reviews"].astype(int)



data["Reviews"]
data["Size"] =data["Size"].astype(str).str.replace('Varies with device', "0")

data["Size"] = data["Size"].astype(str).str.replace("M","")

data["Size"] = data["Size"].str.replace(",","")

data["Size"] = data["Size"].str.replace("+","")

data["Size"] = data["Size"].astype(str).str.replace("k","").astype(float)*1024
data["Size"]
data["Installs"]=data["Installs"].astype(str).str.replace("+", "")

data["Installs"]=data["Installs"].astype(str).str.replace(",","")

data["Installs"]=data["Installs"].astype(str).str.replace('Free',"")

data["Installs"] = data["Installs"].astype(int)

data["Installs"]
data["Price"] = data["Price"].astype(str).str.replace('Everyone',"0")

data["Price"] = data["Price"].astype(str).str.replace("$","")

data["Price"] = data["Price"].astype(float)

data["Price"] = data["Price"]*70

data["Price"].unique()
#now i am using such of lib for visualisation

%matplotlib inline

sns.heatmap(data.corr(),cmap='coolwarm')
import plotly

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
number_of_apps_in_category = data['Category'].value_counts().sort_values(ascending=True)



df = [go.Pie(labels = number_of_apps_in_category.index,values = number_of_apps_in_category.values,hoverinfo = 'label+value')]



plotly.offline.iplot(df, filename='active_category')
df = [go.Histogram(

        x = data.Rating,

        xbins = {'start': 1, 'size': 0.1, 'end' :5}

)]



print('Average app rating = ', np.mean(data['Rating']))

plotly.offline.iplot(df, filename='overall_rating_distribution')
#most reviewed app rating

plt.figure(figsize=(12,6))

sns.distplot(data["Rating"],bins=10,color="red")
plt.figure(figsize=(12,6))

sns.barplot(x = data.groupby('Category')['Rating'].mean().index, y = data.groupby('Category')['Rating'].mean().values)

plt.xlabel('Category', fontsize=13)

plt.ylabel('Rating', fontsize=13)

plt.xticks(rotation=90)

plt.title("avg rating table based on category")
most_popular_apps = data[(data["Reviews"]>10000000) ][ (data["Rating"]>=4.5)]

sns.countplot(most_popular_apps["Category"])

plt.xticks(rotation=90)
sns.set_context('talk',font_scale=1)

plt.figure(figsize=(17,13))

sns.countplot(data=data,y="Category",hue="Type")
# Box plot 
plt.figure(figsize=(16,12))

sns.boxplot(data=data,x="Size",y="Category",palette='rainbow')
sns.countplot(x=data["Type"])
plt.figure(figsize=(17,13))

sns.countplot(data=data[data["Reviews"]>1000000],y="Category",hue="Type")

plt.title("most popular apps with 1000000+ reviews")

plt.xlabel("no of apps")


plt.figure(figsize=(12,6))

sns.distplot(data[data["Reviews"]>10000]["Rating"],bins=10,color="red")
plt.figure(figsize=(16,6))

sns.scatterplot(data=data[data["Reviews"]>100000],x="Size",y="Rating",hue="Type")

plt.title("apps with reviews graterthan 100000")
x=np.log(data["Installs"])

y=np.log(data["Reviews"])

popular_apps = data[(data["Installs"]>10000000) & (data["Rating"]>=4.7)]



pd.DataFrame(popular_apps[popular_apps["Type"]=="Free"][["App"]])
from sklearn import metrics

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
mldata = data[["Reviews","Size","Installs","Price","Rating"]]

mldata.dropna(inplace=True)



X=mldata.iloc[:,0:-1].values

y = mldata.iloc[:,-1].values
xtrain,xtest,ytrain,ytest = train_test_split(X,y)
#Fit regressor or model on data

rfr = RandomForestRegressor(n_estimators=300)
rfr.fit(xtrain,ytrain)

ypre = rfr.predict(xtest)



df=pd.DataFrame()



df["ytest"]=pd.Series(ytest)



df["ypre"] =pd.Series(ypre)

df.sample(10)
import collections

count = 1

for i in data['Category'].unique():

    print(count,': ',i)

    count = count + 1



sns.set_style('whitegrid')

plt.figure(figsize=(16,8))

plt.title('Number of apps on the basis of category')

sns.countplot(x='Category',data = data)

plt.xticks(rotation=90)

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(15,8))

sns.scatterplot(y='Category',x='Reviews',data = data,hue='Category',legend=False)

plt.xticks(rotation=90)

plt.title('Number of reviews on the basis of Category')

plt.show()
plt.figure(figsize=(20,8))

data.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(10).plot(kind='bar');

plt.ylabel('Count', fontsize=16)

plt.xlabel('Ratings', fontsize=16)

plt.title("Total Reviews Number for Top 10 Category", fontsize=16)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25,10))

plt.scatter(x=data["Genres"],y=data["Rating"],color="green",marker="o")

plt.xticks(rotation=90)

plt.grid()

plt.show()





