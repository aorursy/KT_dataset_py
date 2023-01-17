import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path

data = Path('/kaggle/input/covid19-in-india')



os.listdir(data)
df=pd.read_csv(data/'IndividualDetails.csv',index_col='detected_state')

df.head()
KL = pd.DataFrame(df.loc["Kerala"])

KL.tail()
KL.drop('government_id',axis=1,inplace=True)
KL.head()
KL.drop(['id','age','gender','detected_city','notes'],axis=1,inplace=True)
KL.isnull().sum()

KL.drop('nationality',axis=1,inplace=True)

KL.head()
rec=0

hosp=0

Dec=0

current=[]

for i in KL['current_status']:

    if i == 'Recovered':

        rec +=1

    elif i == 'Hospitalized':

        hosp +=1

    else:

        Dec +=1

print(hosp , rec, Dec)

current.append(hosp)

current.append(rec)

current.append(Dec)

print(current)



plt.figure(figsize=(10,8))

plt.bar(KL['current_status'],KL['diagnosed_date'],width=0.1)

KL['current_status'].value_counts().plot(kind='bar',

                                         color=['red','green','black'],width=0.1)

plt.grid()
plt.figure(figsize=(20,8))

plt.bar(KL['detected_district'],KL['diagnosed_date'],width=0.2)

plt.scatter(KL['detected_district'],KL['diagnosed_date'])

plt.grid()
plt.figure(figsize=(20,5))

plt.bar(KL['detected_district'],KL['current_status'],width=0.2)

plt.scatter(KL['detected_district'],KL['current_status'])