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
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/daily_covid19_data_tr.csv")
df.tail()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.Active_Cases,use_line_collection=True)



plt.title('Active Cases', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)





plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 100000, 10000),size=20)





plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.daily_new_cases, use_line_collection=True)



plt.title('Daily New Cases', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)





plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 6000, 500),size=20)







plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.daily_new_recovered, use_line_collection=True)



plt.title('Daily New Recovered', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(13, 56)



plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 6000, 500),size=20)







plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.intensive_care_number,use_line_collection=True)



plt.title('Active Intensive Care Patients', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(14, 56)



plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 2250, 250),size=20)







plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.intubated_care_number, use_line_collection=True)



plt.title('Active Intubated Care Patients', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(15, 56)



plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 1400, 200),size=20)





plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.Recovered,use_line_collection=True)



plt.title('Total Recovered', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(15, 56)



plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 80000, 5000),size=20)







plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.Deaths, use_line_collection=True)



plt.title('Total Deaths', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)





plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 4500, 500),size=20)







plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.daily_new_deaths, use_line_collection=True)



plt.title('Daily New Deaths', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(4, 56)



plt.xticks(size=10,rotation=90)

plt.yticks(np.arange(0, 140,25),size=20)





plt.show()
plt.figure(figsize=(16, 9))



plt.stem(df.date, df.tested_person_number, use_line_collection=True)



plt.title('Total Tested', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)





plt.xticks(size=10, rotation=90)

plt.yticks(np.arange(0, 1400000, 100000),size=20)





plt.show()

plt.figure(figsize=(16, 9))



plt.plot(df.date, df.Active_Cases,label="Active Cases", marker="o")

plt.plot(df.date, df.Recovered, label="Recovered", marker = "o")



plt.title('Active Cases & Total Recovered', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(5, 56)



plt.legend(loc=2, ncol=10)



plt.xticks(size=10, rotation=90)







plt.show()
plt.figure(figsize=(16, 9))



plt.plot(df.date, df.intubated_care_number, marker= "o", label="Active Intubated",)

plt.plot(df.date, df.intensive_care_number, marker= "o", label="Active Intensive Care",)



plt.title('Active Intubated & Active Intensive Care Patient', size=30)



plt.xlabel('Date', size=20)

plt.ylabel('# of People', size=20)



plt.xlim(14, 56)



plt.legend(loc=2, ncol=10)



plt.xticks(size=10,rotation=90)

plt.yticks(size=20)



plt.show()