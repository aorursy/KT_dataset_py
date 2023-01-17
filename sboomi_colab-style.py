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
por = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-por.csv")

por
#Load Students learning Maths as a language csv file

mat = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")

mat
#Merge students studying both portuguese and math

result = por.merge(mat[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]], how = 'inner',indicator=False)

result
#Weekly Consumed alcohol as per father's job

import matplotlib.pyplot as plt

import seaborn as sns

fatherJobsList = list(result.Fjob.unique())

FstudentDalc = []

for each in fatherJobsList:

    x = result[result.Fjob == each]

    FstudentDalc.append(sum(x.Dalc))



#sorting

sort_data = pd.DataFrame({'father_jobs':fatherJobsList,'student_dalc':FstudentDalc})

new_index3 = (sort_data['student_dalc'].sort_values(ascending=False)).index

sorted_data3 = sort_data.reindex(new_index3)



#Visualitizon    

plt.figure(figsize=(15,15))

sns.barplot(x=FstudentDalc,y=fatherJobsList)

plt.xticks(rotation=90)

plt.xlabel("Weekly Alcohol Consumption")

plt.ylabel("Father Jobs")

plt.show()
#Number of Students from low to high consumer on worday and weekend



#marker - big dot

import matplotlib.pyplot as plt

l=[1,2,3,4,5] #Alcohol consumption levels from 1 - very low to 5 - very high

labels= "1-Very Low","2-Low","3-Medium","4-High","5-Very High"

plt.figure(figsize=(15,5))

plt.plot(labels,list(map(lambda l: list(result.Dalc).count(l),l)),color="red",linestyle="--",marker="o", markersize=10,label="Workday")

plt.plot(labels,list(map(lambda l: list(result.Walc).count(l),l)),color="green",linestyle="--",marker="o", markersize=10,label="Weekend")

plt.title("Student Alcohol Consumption")

plt.grid()

plt.ylabel("Number of Students")

plt.legend()

plt.show()