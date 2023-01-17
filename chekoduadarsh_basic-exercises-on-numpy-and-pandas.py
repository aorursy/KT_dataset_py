import numpy as np

import pandas as pd
arr = np.arange(10)

arr
arr[arr % 2 == 1] = -1

arr
arr = np.arange(10)

arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols
a =  np.array([2,4,6])

np.r_[np.repeat(a, 3), np.tile(a, 3)]
a = np.array([1,2,3,9,3,4,3,4,5,6])

b = np.array([10,6,5,7,11,12,4,5,9,66])

np.intersect1d(a,b)
a = np.array([1,2,3,9,3,4,3,4,5,6])

b = np.array([10,6,5,7,11,12,4,5,9,66])

np.setdiff1d(a,b)
arr = np.arange(9).reshape(3,3)

arr = arr[[1,0,2], :]

arr
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)

iris_1d
petal_length = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2]) #usecols is number of coloumn

mu, med, sd = np.mean(petal_length), np.median(petal_length), np.std(petal_length)

print("\nMean:",mu,"\nMedian", med,"\nStandard Deviation", sd)
petal_length = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2]) #usecols is number of coloumn

pmax, pmin = petal_length.max(), petal_length.min()



p = (petal_length - pmin)/(pmax - pmin)



p
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

np.isnan(iris).any()

iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

iris[np.isnan(iris)] = 0

iris
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

iris_list = iris.tolist()

iris_list
data = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

iris_df = pd.DataFrame(data=data[1:,1:],    # values

            index=data[1:,0],    # 1st column as index

           columns=data[0,1:])  # 1st row as the column names

iris_df
from PIL import Image

import numpy as np



w, h = 512, 512

data = np.zeros((h, w, 3), dtype=np.uint8)

data[0:256, 0:256] = [255, 0, 0] # red patch in upper left

img = Image.fromarray(data, 'RGB')



img
score = np.array([70, 60, 50, 10, 90, 40, 80])

name = np.array(['Ada', 'Ben', 'Charlie', 'Danny', 'Eden', 'Fanny', 'George'])

sorted_name = name[np.argsort(score)] # an array of names in ascending order of their scores

sorted_name
df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9]})

df


df["Col2"]= [10, 20, 30,20,30,40,50,60,80,70,90]

df
df.head(10)
df.tail(10)
iris_df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")

iris_df
import pandas_profiling 

titanic_df = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_df_p = titanic_df.profile_report()

titanic_df_p
df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})

df.rename(columns={"col1": "COL1", "col2": "COL2"})
df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})

df2 = pd.DataFrame({"col1":[10] , "col2":[100]})

df.append(df2)
df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})

df2 = pd.DataFrame({"col1":[10] , "col2":[100]})

df.append(df2, ignore_index=True)
date_from = "2019-01-01"

date_to = "2019-01-12"

date_range = pd.date_range(date_from, date_to, freq="D")

date_range
left = pd.DataFrame({"key": ["key1", "key2", "key3", "key4"], "value_l": [1, 2, 3, 4]})

left
right = pd.DataFrame({"key": ["key3", "key2", "key1", "key6"], "value_r": [3, 2, 1, 6]})

right
df_merge = left.merge(right, on='key', how='left', indicator=True)

df_merge
iris_df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")

filtered_iris_df = iris_df[iris_df.species.isin(['Iris-setosa','Iris-virginica'])] 

filtered_iris_df
df1 = pd.DataFrame({'city':['new york','mumbai','paris'] , 'temp_windspeed': [[21,4],[34,3],[26,5]]})

df1
df2 = df1.temp_windspeed.apply(pd.Series)

df2.rename(columns= {'0':'temperature','1':'windspeed'})

df2