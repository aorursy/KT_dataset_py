# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/2018-chicago-employees-salaries/chicago_employees.csv')
df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
df.drop('Hourly Rate',axis=1, inplace=True)
#for i in df["Annual Salary"]:

    #print(type(i))
for i in range(df["Annual Salary"].size):

    if(type(df["Annual Salary"][i])==str):

            df["Annual Salary"][i]=df["Annual Salary"][i].replace("$","")

            df["Annual Salary"][i]=df["Annual Salary"][i].replace(".00","")





df["Annual Salary"].head(10)
for i in range(df["Annual Salary"].size):

    if(type(df["Annual Salary"][i])==str):

        if(not df["Annual Salary"][i]==False):

            df["Annual Salary"][i]=float(df["Annual Salary"][i])
df["Annual Salary"].plot(kind='hist',title="Annual Salary Histogram with null values included")
df["Annual Salary"].plot.kde()
df["Annual Salary"].describe()
#median=df["Annual Salary"].median()

#value1={"Annual Salary": median}

#df2=df.fillna(value=value1)

mean=df["Annual Salary"].mean()

value2={"Annual Salary": mean}

df2=df.fillna(value=value2)

#ALREADY RUNED AND COPPIED



df3["Annual Salary"].plot(kind='hist',title="Histogram of Annual Salary after filling nulls with mean")
df2["Typical Hours"].value_counts()
df1= df2.drop("Typical Hours", axis=1, inplace=False)
df1["Full or Part-Time"].value_counts()
df["Department"].value_counts()
df["Job Titles"]
police=df1[df1["Department"]=="POLICE"]

#now lets see how much Finance people make

finance=df1[df1["Department"]=="FINANCE"]
it=df1[df1["Department"]=="DoIT"]
it["Annual Salary"].describe()
police["Annual Salary"].describe()
finance["Annual Salary"].describe()
df1["Annual Salary"].describe()
police["Annual Salary"].mean()
police["Annual Salary"].std()
it["Annual Salary"].plot.kde(title="IT section KDE fitted plot / mean null values")
finance["Annual Salary"].plot.kde(title="Finance section KDE fitted plot/ mean null values")
police["Annual Salary"].plot.kde(title='Police section KDE fitted plots/ mean null values')
df1["Annual Salary"].plot.kde(title='All salaries KDE fitted plots/ mean null values')
police["Annual Salary"].plot.kde(bw_method=0.4)
df["Annual Salary"].plot.kde(bw_method=0.2)
out1=df[df["Annual Salary"]>180000]

out2=df[df["Annual Salary"]<30000]
out1.to_csv("best.csv")
out2.to_csv("worst.csv")
x=df[df["Department"]=="POLICE"]
x
x["Annual Salary"].plot.kde()