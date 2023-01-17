# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/coronavirus-cases-in-india/Covid cases in India.csv')
df.head()
x=df['Name of State / UT']
xpos=np.arange(len(x))
y=df['Total Confirmed cases']
plt.bar(xpos,y,color=['pink','lightcoral','violet','gold','lightskyblue'])
plt.xticks(xpos,x)
plt.xticks(rotation=90)
plt.ylabel('Count')
cured=df[df['Cured/Discharged/Migrated']==True]
deaths=df[df['Deaths']==True]
nos=[cured['Cured/Discharged/Migrated'].sum(),deaths['Deaths'].sum()]
labels=['Cured','Deaths']
plt.pie(nos,labels=labels,startangle=90,shadow=True)

plt.subplot(1,2,1)
df['Cured/Discharged/Migrated'].value_counts().plot.bar(color='green')
plt.title('Cured rate',size=15)
plt.xlabel('Cures in a week')
plt.ylabel('Count')
plt.show()


plt.subplot(1,2,2)
df['Deaths'].value_counts().plot.bar(color='red')
plt.title('Death rate',size=15)
plt.xlabel('Deaths in a week')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(8,8))
cure=df['Cured/Discharged/Migrated'].sum()
death=df['Deaths'].sum()

tot=df['Total Confirmed cases'].sum()
tot=tot-cure-death
act=[tot,cure,death]
lab=['Total cases','Cured cases','Deaths']
colors=['lightcoral','green','red']
explode=[0.1,0.1,0.1]
plt.pie(act,labels=lab,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)
plt.title('Overall view of COVID-19',fontsize=20)