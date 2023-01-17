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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = 30
data=pd.read_csv('../input/homicide-reports/database.csv')
data
from sklearn.preprocessing import LabelEncoder
lmonths=LabelEncoder()
data['Month']=lmonths.fit_transform(data['Month'])
data['Month']=lmonths.inverse_transform(data['Month'])
data[data['Perpetrator Race']=='Unknown']
sns.distplot(data['Year'])
yr=data[data['Crime Solved']=='Yes']['Year'].value_counts(sort=False)
yr=pd.DataFrame(yr)
yr['Yearno']=yr.index
yr['Year Count']=yr['Year']
Nr=data[data['Crime Solved']=='No']['Year'].value_counts(sort=False)
Nr=pd.DataFrame(Nr)
Nr['Yearno']=Nr.index
Nr['Year Count']=Nr['Year']
Yer=data['Year'].value_counts(sort=False)
Yer=pd.DataFrame(Yer)
Yer['Yearno']=Yer.index
Yer['Year Count']=Yer['Year']
sns.set(rc={'figure.figsize':(20,10)})
sns.barplot(x='Yearno',y='Year Count',data=Yer,saturation=1,color='y',label='Total Crime')
sns.barplot(x='Yearno',y='Year Count',data=yr,saturation=0.5,color='b',label='Total Crime Solved')
sns.barplot(x='Yearno',y='Year Count',data=Nr,color='r',label='Total Crimes Unsolved')
plt.legend()
data['Agency Type'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data['Crime Type'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data['Victim Ethnicity'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data['Perpetrator Sex'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data['Weapon'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data['Victim Race'].value_counts().plot(kind='barh',color=['r','g','b','c','y','brown'])
data
mn=pd.DataFrame(data['Month'].value_counts(sort=False))
mn
mn['Monthname']=mn.index
mn['Month total Cases']=mn['Month']
sns.barplot(x='Monthname',y='Month total Cases',data=mn,saturation=1,color='b',label='Total Crime')
