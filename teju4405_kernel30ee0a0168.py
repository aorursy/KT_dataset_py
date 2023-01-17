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
province=pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')
province.head()
province['Month']=pd.DatetimeIndex(province['date']).month
province['Month']=province['Month'].apply({1:'January', 2:'February',3:'March',4:'April',5:'May',6:'June'}.get)
province.head()
province.province.unique()
province.info()
from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
plt.bar(province['province'],province['confirmed'],linewidth=5,color='red')
ax=plt.gca()
for x in ax.xaxis.get_ticklabels():
    x.set_rotation(90)
plt.legend(['Confirmed'],loc='best',frameon=False)
plt.title('Covid Variation from January to June')

plt.figure(figsize=(5,5))
plt.bar(province['province'],province['released'],linewidth=5,color='violet')
ax=plt.gca()
for x in ax.xaxis.get_ticklabels():
    x.set_rotation(90)
plt.legend(['Released'],loc='best',frameon=False)
plt.title('Covid Variation from January to June')

plt.figure(figsize=(5,5))
plt.bar(province['province'],province['deceased'],linewidth=5,color='pink')
ax=plt.gca()
for x in ax.xaxis.get_ticklabels():
    x.set_rotation(90)
plt.legend(['Deceased'],loc='best',frameon=False)
plt.title('Covid Variation from January to June')

Seoul=province[province['province']=='Seoul']
Seoul.head()
plt.figure(figsize=(5,5))
plt.plot(Seoul['Month'],Seoul['released'],linewidth=3,color='yellow')
plt.bar(Seoul['Month'],Seoul['confirmed'],linewidth=5,color='red')
ax=plt.gca()

for x in ax.xaxis.get_ticklabels():
    x.set_rotation(90)
plt.legend(['Released','Confirmed'],loc='best',frameon=False)
plt.title('Covid Variation from January to June in Seoul')

Daegu=province[province['province']=='Daegu']
Daegu.head()
plt.figure(figsize=(5,5))
plt.plot(Daegu['Month'],Daegu['released'],linewidth=3,color='green')
plt.bar(Daegu['Month'],Daegu['confirmed'],linewidth=5,color='violet')
ax=plt.gca()

for x in ax.xaxis.get_ticklabels():
    x.set_rotation(90)
plt.legend(['Released','Confirmed'],loc='best',frameon=False)
plt.title('Covid Variation from January to June in Daegu')

S_D=pd.merge(Seoul,Daegu,how='outer',left_index=False,right_index=False)
S_D.head()
plt.plot(S_D['province'],S_D['released'],linewidth=2,color='pink')
plt.bar(S_D['province'],S_D['confirmed'],linewidth=2,color='green')
plt.legend(['Released','Confirmed'])
gender_cases=pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')
gender_cases.head()
gender_cases['Month']=pd.DatetimeIndex(gender_cases['date']).month
gender_cases['Month']=gender_cases['Month'].apply({1:'January', 2:'February',3:'March',4:'April',5:'May',6:'June'}.get)
gender_cases.head()
plt.figure(figsize=(5,5))
plt.bar(gender_cases['sex'],gender_cases['confirmed'],linewidth=5,color='violet')
plt.legend(['Confirmed'],loc='best',frameon=False)
plt.title('Confirmed Cases in South Korea')

plt.figure(figsize=(5,5))
bars=plt.bar(gender_cases['sex'],gender_cases['deceased'],linewidth=3,color='green')
plt.legend(['Deceased'],loc='best',frameon=False)
plt.title('Deceased Cases in South Korea')