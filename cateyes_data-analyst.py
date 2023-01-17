# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")

data.head(3)
data.info()
data.describe()
data.columns=[ each.lower() for each in data.columns]

data.columns=[each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]

data.columns
data1=data.job_title.value_counts()



data1=pd.DataFrame(data1).reset_index()



data11=data1.head(15)



data11
plt.figure(figsize=(12,6))



plt.bar(data11['index'], data11['job_title'],color='orange')



plt.xticks(rotation=90)



plt.xlabel('Jobs')



plt.ylabel('Count')



plt.title("Count of Jobs")



plt.show()
data2=data.salary_estimate.value_counts()



data2=pd.DataFrame(data2).reset_index()



data21=data2.head(15)



data21
plt.figure(figsize=(12,6))



plt.bar(data21['index'], data21['salary_estimate'],color='yellow')



plt.xticks(rotation=90)



plt.xlabel('Salary')



plt.ylabel('Count')



plt.title("Count of Salary")



plt.show()
data.job_title=[ each.lower() for each in data.job_title]

data.job_title=[each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.job_title]

data.job_title
data_analyst= data[data.job_title== "data_analyst"]

data31=data_analyst.salary_estimate.value_counts()

data32=pd.DataFrame(data31).reset_index()

data33=data32.head(10)
data33.plot(kind="scatter",y="index",x="salary_estimate",color="red")



plt.xlabel("Count of Data Analyst")



plt.ylabel("Salary Estimate")



plt.show()
data4=data_analyst.location.value_counts()

data41=pd.DataFrame(data4).reset_index().head(15)

data41
data41.plot(kind="scatter",y="index",x="location",color="purple")

plt.xlabel("Count of Location ")

plt.ylabel("Names of Location")

plt.title("Locations and Data Analyst")

plt.show()