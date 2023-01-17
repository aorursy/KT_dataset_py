# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sys

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv('/kaggle/input/sales-data/sales_data_sample.csv')

data.drop(["ADDRESSLINE1","ADDRESSLINE2","STATE","TERRITORY"],axis=1,inplace=True)

data["ORDERDATE"]=pd.to_datetime(data.ORDERDATE)

data.index=data["ORDERDATE"]

data['TITLE'] = data['CUSTOMERNAME'].str.extract(' ([A-Za-z]+)\.')

data.drop("ORDERDATE",axis=1,inplace=True)

data = data.sort_index()
data.head()
data.info()
data.isnull().sum()
data.CITY.value_counts()
data.TITLE.value_counts()
sns.heatmap(data.isnull())
sns.countplot(x="CITY",hue="DEALSIZE",data=data)

plt.legend(loc="upper right")

plt.ylabel("COUNT")

plt.xticks(rotation=60)

plt.show()
sns.countplot(x="TITLE",hue="DEALSIZE",data=data)

plt.ylabel("COUNT")

plt.show()
sns.countplot(x="COUNTRY",hue="DEALSIZE",data=data)

plt.legend(loc="upper right")

plt.ylabel("COUNT")

plt.xticks(rotation=60)

plt.show()
sns.countplot(x="STATUS",data=data)

plt.ylabel("COUNT")

plt.show()
monthly_revenue = data.groupby(['YEAR_ID','MONTH_ID'])['SALES'].sum().reset_index()

monthly_revenue2003 = monthly_revenue.loc[monthly_revenue.YEAR_ID==2003]

monthly_revenue2003.index = np.arange(1,len(monthly_revenue2003)+1)

monthly_revenue2004 = monthly_revenue.loc[monthly_revenue.YEAR_ID==2004]

monthly_revenue2004.index = np.arange(1,len(monthly_revenue2004)+1)

monthly_revenue2005 = monthly_revenue.loc[monthly_revenue.YEAR_ID==2005]

monthly_revenue2005.index = np.arange(1,len(monthly_revenue2005)+1)

plt.plot(monthly_revenue2003.SALES,label="2003")

plt.plot(monthly_revenue2004.SALES,label="2004")

plt.plot(monthly_revenue2005.SALES,label="2005")

plt.xticks(np.arange(1,13))

plt.legend()

plt.title("Monthly Revenue")

plt.xlabel("Months")

plt.ylabel("Revenue")

plt.show()
yearly_revenue = data.groupby(['YEAR_ID'])['SALES'].sum().reset_index()

plt.bar(yearly_revenue.YEAR_ID, yearly_revenue.SALES, label="Revenue", color='c', alpha=0.7)

plt.xticks([2003,2004,2005])

plt.title("Yearly Revenue")

plt.xlabel("Years")

plt.ylabel("Revenue")

plt.text(s="5 months of data",x=2004.8,y=1000000)

plt.legend()

plt.rcParams["xtick.labelsize"]=11

plt.show()
sns.countplot(x="ORDERLINENUMBER",data=data)

plt.ylabel("COUNT")

plt.rcParams["xtick.labelsize"]=11

plt.show()
sns.countplot(x="CONTACTLASTNAME",data=data)

plt.xticks(rotation=60)

plt.ylabel("COUNT")

plt.rcParams["xtick.labelsize"]=11

plt.show()
sns.countplot(x="CONTACTFIRSTNAME",data=data)

plt.ylabel("COUNT")

plt.xticks(rotation=60)

plt.rcParams["xtick.labelsize"]=11

plt.show()
plt.subplot(1,2,1)

sns.distplot(data["PRICEEACH"])

plt.subplot(1,2,2)

sns.distplot(data["QUANTITYORDERED"])

plt.show()
plt.subplot(2,2,1)

sns.distplot(data.loc[data["TITLE"]=="Co"]["SALES"])

plt.title("Sales for Co")

plt.subplot(2,2,2)

sns.distplot(data.loc[data["TITLE"]=="Inc"]["SALES"])

plt.title("Sales for Inc")

plt.subplot(2,2,3)

sns.distplot(data.loc[data["TITLE"]=="Ltd"]["SALES"])

plt.title("Sales for Ltd")

plt.subplot(2,2,4)

sns.distplot(data.loc[data["TITLE"]=="Corp"]["SALES"])

plt.title("Sales for Corp")

plt.tight_layout()

plt.show()
sns.violinplot(x="TITLE",y="SALES",data=data)

plt.title("Sales")

plt.show()
data.dropna(inplace=True)
status = pd.get_dummies(data.STATUS)

city = pd.get_dummies(data.COUNTRY)

country = pd.get_dummies(data.CITY)

firstname = pd.get_dummies(data.CONTACTFIRSTNAME)

lastname = pd.get_dummies(data.CONTACTLASTNAME)

title = pd.get_dummies(data.TITLE)

data1= data.drop(["ORDERNUMBER","STATUS","COUNTRY","CITY","CONTACTFIRSTNAME","CONTACTLASTNAME","DEALSIZE",

                  "TITLE","POSTALCODE","PHONE","CUSTOMERNAME","PRODUCTCODE","PRODUCTLINE"],axis=1)

data2 = pd.concat([data1,status,city,country,firstname,lastname,title],axis=1)
data2.head()
from sklearn.cluster import KMeans
model = KMeans( n_clusters=3)

model.fit(data2)
cls = model.labels_
plt.scatter(data1.PRICEEACH,data1.SALES,c=cls)

plt.show()
pd.Series(cls).value_counts()
data.DEALSIZE.value_counts()
X = data2

y = data.DEALSIZE
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=42)
model2 = tree.DecisionTreeClassifier()

model2.fit(X_train,y_train)
model2.score(X_test,y_test)
pred = model2.predict(X)

accuracy_score(y,pred)
confusion_matrix(y,pred)
for i in range(len(pred)):

    if pred[i]=='Small':

        pred[i]=0

    elif pred[i]=='Medium':

        pred[i]=1

    elif pred[i]=='Large':

        pred[i]=2
plt.scatter(data1.PRICEEACH,data1.SALES,c=pred)

plt.show()