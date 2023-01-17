import numpy as np

import pandas as pd

from pandas import DataFrame,Series
data=pd.read_csv('../input/BlackFriday.csv')
data
data.columns
len(data.columns)
data.count(axis='columns')
data.index
data.tail()
data.tail(3)
data.tail(10)
data.head()
data.head(3)
data.head(8)
#groupby()

#--> it will output the result on the basis of a certain column

data.groupby(['Product_ID']).count()
data.groupby(['Marital_Status']).count()
v1=sum(data.Marital_Status==1)

v1
v2=sum(data.Marital_Status==0)

v2
varTotal=v1+v2

varTotal
def DataInPercent (DataInNumber,Total):

    percent=(DataInNumber/Total)*100

    return percent
print("Married persons:",round(DataInPercent(v1,varTotal),2),"%")
print("UnMarried persons:",round(DataInPercent(v2,varTotal),2),"%")
sum(data.Age<='18')
sum(data.Age>='18')
data[data.Marital_Status==1]
data[data.Marital_Status==0]
print("Male Percentage:",round(DataInPercent(sum(data.Gender=='M'),537577)),"%")
print("Female Percentage:",round(DataInPercent(sum(data.Gender=='F'),537577)),"%")