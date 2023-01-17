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
import pandas as pd 

import matplotlib.pyplot as plt

data=pd.read_csv("../input/world-university-rankings/timesData.csv")

data.head()



# Cleaning data



data.drop(["female_male_ratio","student_staff_ratio"],axis=1,inplace=True)



data=data[data["total_score"]!="-"]



#data=data.dropna(how="all")



data["total_score"]=pd.to_numeric(data["total_score"])



data.head()



# What is the best 20 university in 2016?



data[(data["total_score"]>=82) & (data["year"] ==2016)].head(20)
# What is the best 20 university in 2014?



data[(data["total_score"]>=70) & (data["year"] ==2014)].head(20)

#What is the most succesful contries ?



plt.style.use("ggplot")

ülkeler = data['country'].value_counts().keys().tolist()

counts = data['country'].value_counts().tolist()

plt.figure(figsize=(15,10))

plt.xlabel("Countries",weight="bold")

plt.ylabel("Number of universities",weight="bold")

plt.title("Most succesful countries in all years",weight="bold")

plt.bar(ülkeler,counts)

plt.xticks(ülkeler,rotation="vertical",size=9,weight="bold")

plt.show()





# what is the number of students first 20 universities in 2016  ?



best=data[(data["total_score"]>=82) & (data["year"] ==2016)].head(20)



num_student=best["num_students"].tolist()

dizi=[]

for i in num_student:

    i=i.replace(",","")

    i=int(i)

    dizi.append(i)





liste=best["university_name"].tolist()



plt.figure(figsize=(15,10))

plt.bar(liste,dizi)

plt.xticks(liste,rotation="vertical",size=10,weight="bold")

plt.title("Total number of students",weight="bold")

plt.show()
#Which Turkish universities are in the ranking?

data[data["country"]=="Turkey"]
