# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df0 = pd.read_csv("../input/0.csv",header = None)

df1 = pd.read_csv("../input/1.csv")

df2 = pd.read_csv("../input/2.csv")

df3 = pd.read_csv("../input/3.csv")

df0.head()
col_names = list()

for i in range(0,65):

    if i == 64:

        col_names.append("class")

    else:

        col_names.append("sensor"+str(i+1))

    
df0.columns = col_names

df1.columns = col_names

df2.columns = col_names

df3.columns = col_names
print(df0.shape)

print(df1.shape)

print(df2.shape)

print(df3.shape)
df = pd.concat([df0,df1,df2,df3],ignore_index=True)

print(df.tail())

print(df.shape)
df.info()
total = df.isnull().sum().sort_values(ascending = False)

percentage = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total,percentage],axis = 1, keys = ["Total","Percentage"])

missing_data
print(df.shape)

df.dropna(how = "any", inplace = True)

print(df.shape)
sensor1 = df.sensor1.unique()

len(sensor1)
#For special columns

df.drop_duplicates(subset=[col_name],keep = "first",inplace = True)
print(df.shape)

df.drop_duplicates(keep = "first",inplace = True)

print(df.shape)
df.describe()
#boxplot

import seaborn as sns

sns.boxplot(x = df['sensor32'])
#Z score

def detect_outlier(data):

    outliers=[]

    

    threshold=3

    mean = np.mean(data)

    std  = np.std(data)

    

    

    for i in data:

        z_score= (i - mean)/std         #z = (x-mean)/std

        if np.abs(z_score) > threshold:

            outliers.append(i)

    return outliers
outlier = detect_outlier(df.sensor1)

len(outlier)
#Compute number of outliers for all columns

number_outliers = []

x = np.arange(1,66,1) 



for i in df.columns:

    outliers = detect_outlier(df[i])

    c = len(outliers)

    number_outliers.append(c)

plt.plot(x,number_outliers)

plt.show()
#ıqr (q3-q1) ---> aslında boxplot gibi oluyor.

def detect_outlier2(data):

    outliers = []

    

    data = sorted(data)

    q1, q3 = np.percentile(data,[25,75])

    iqr = q3 - q1

    

    lower_bound = q1 - (1.5 * iqr) 

    upper_bound = q3 + (1.5 * iqr)

    

    for i in data:

        if lower_bound <= i <= upper_bound:

            continue

        else:

            outliers.append(i)

    

    return outliers
outlier = detect_outlier2(df.sensor1)

len(outlier)
number_outliers2 = []

x = np.arange(1,66,1) 



for i in df.columns:

    outliers = detect_outlier2(df[i])

    c = len(outliers)

    number_outliers2.append(c)

plt.plot(x,number_outliers2)

plt.xlabel("Columns")

plt.ylabel("Number of Outliers")

plt.show()
#Z-score

from scipy import stats



z = np.abs(stats.zscore(df))

print(z)
threshold = 3

print(np.where(z > threshold))
zdata = df[(z <= 3).all(axis=1)]

print(zdata.shape)
#next step is to standardize our data - using MinMaxScaler

y = df["class"]

x_data1 = df.drop(["class"],axis = 1)



from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()

scaler.fit(x_data1)



x_data1 = pd.DataFrame(scaler.transform(x_data1), index=x_data1.index, columns=x_data1.columns)

x_data1.iloc[4:10]
#other way for normalization

y = df["class"]

x_data2 = df.drop(["class"],axis = 1)



x = (x_data2 - np.min(x_data2)) / (np.max(x_data2) - np.min(x_data2)).values

x_data2 = pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)

x_data2.iloc[4:10]

sns.countplot(x='class', data=df)
#0 and 1lerin indeksleri

count = 0

for i in y:

    if i == 0 or i == 1:

        count+=1

    else:

        break

print(count)
y = y[0:5812].values.reshape(-1,1)

x = x_data1[0:5812].values



print(y.shape)

print(x.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42) 
a=0

b=0

c=0

d=0

for i in y_train:

    if i ==0:

        a+=1

    else:

        b+=1

for i in y_test:

    if i ==0:

        c+=1

    else:

        d+=1

    

print("0 class for train: ",a,"\n1 class for train: ",b,"\n0 class for test: ",c,"\n1 class for test: ",d)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)



print("Test Accuracy {}".format(log_reg.score(x_test,y_test)))
x5 = df.drop(["class"],axis=1)

x5 = x5.values

y5 = df["class"].values.reshape(-1,1)



x5 = x5[0:5812]

y5 = y5[0:5812]
from sklearn.model_selection import train_test_split

x_train1, x_test1, y_train1, y_test1 = train_test_split(x5,y5,test_size = 0.3, random_state = 42) 
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train1,y_train1)



print("Test Accuracy: {}".format(lr.score(x_test1,y_test1)))
sns.countplot(x="class",data=zdata)
y6 = zdata["class"].values.reshape(-1,1)

y6[2846]
x6 = zdata.drop(["class"],axis=1)

x6 = x6.values
from sklearn.model_selection import train_test_split

x_train2, x_test2, y_train2, y_test2 = train_test_split(x6,y6,test_size = 0.3, random_state = 42) 
from sklearn.linear_model import LogisticRegression



lr2 = LogisticRegression()

lr2.fit(x_train2,y_train2)



print("Accuracy: {}".format(lr2.score(x_test2,y_test2)))
y = df["class"].values.reshape(-1,1)

x = df.drop(["class"],axis = 1).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42) 
#for K=3;

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  #n_neighbor = k

knn.fit(x_train,y_train)



#Accuracy for K=7

print("K={} iken accuracy: {}".format(3,knn.score(x_test,y_test)))
#for K=7;

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)



#Accuracy for K=7

print("K={} iken accuracy: {}".format(3,knn.score(x_test,y_test)))



#Alttaki hatayı almamak için np.ravel(y_train,order="C") yapıyoruz nedenine bak! Alttaki kodda var.

#Another way : model = knn.fit(train_fold, train_y.values.reshape(-1,))

#Açıklaması net bir şekilde burda: https://www.w3resource.com/numpy/manipulation/ravel.php
#Find the best K value

k_value = []

accuracy = []



for i in range(1,22):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,np.ravel(y_train,order='C'))

    

    score = knn.score(x_test,y_test)

    k_value.append(i)

    accuracy.append(score)

for i,j in zip(k_value,accuracy):

    print(i,j)



#Find K value for Max accuracy

plt.plot(range(1,22),accuracy,color = "blue")

plt.xlabel("K value")

plt.ylabel("Accuracy")

plt.show()
"""

#Find max accuracy with range 1-200 for K

max_accuracy = 0



for i in range(1,200):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,np.ravel(y_train,order="C"))

    score = knn.score(x_test,y_test)

    

    if score > max_accuracy:

        k,max_accuracy = i,score

    else:

        continue

print(k,":",max_accuracy)"""
#random_state verme bakalım kendisi sürekli değişsin sonuç ne kadar değişiyor!

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42) 



knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,np.ravel(y_train,order="C"))



print("K=7  Accuracy: {}".format(knn.score(x_test,y_test)))