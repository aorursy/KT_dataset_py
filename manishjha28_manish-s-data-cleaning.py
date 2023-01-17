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
df = pd.read_csv('../input/obesity-among-adults-by-country-19752016/data.csv', index_col=0)
df.head()
df = pd.read_csv('../input/obesity-among-adults-by-country-19752016/data.csv', header=[0,1],skiprows=[1,2],index_col=0)
df.head()
df.columns.names
df.columns.names = ['Year', 'Gender']
df.columns.names
df.index.names
df.index.names = ['Country']
df.index.names
df.head()
df = df.sort_index(axis = 1,level= 0)
df.head()
y = df.stack(level = 0)
y
z = y.reset_index()
z
z = z.melt(id_vars=['Country','Year'], value_vars=['Both sexes', 'Female', 'Male'], value_name='Obesity_levels')
z
z.sort_values(['Country','Year'],inplace = True)

z.reset_index(drop = True)
z['Obesity_levels']=z['Obesity_levels'].apply(lambda x:x.split()[0])
z.reset_index(drop = True)
z = z[z["Obesity_levels"] != "No"]

z['Obesity_levels']=z['Obesity_levels'].apply(lambda x:float(x))
z.reset_index(drop=True)
z.info()
z['Year']=z['Year'].apply(lambda x: int(x))
z.info()
both = z[z["Gender"]=="Both sexes"].groupby("Year").Obesity_levels.mean()
male = z[z['Gender']=="Male"].groupby('Year').Obesity_levels.mean()
female = z[z['Gender']=="Female"].groupby('Year').Obesity_levels.mean()
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(both,linestyle='solid',marker='^',label="Obesity% of both Sexes")
plt.plot(male,linestyle='solid',marker='o',label="Obesity% of Males")
plt.plot(female,linestyle='solid',marker='<',label="Obesity% of Females",color = 'green')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Obesity%', fontsize=20)
plt.title('Mean Obesity by Year', fontsize=20)

plt.grid(True)
plt.legend()
plt.tight_layout()
Most_obese_country = z[((z['Year']==2016)&(z['Gender']=="Both sexes")&(z['Obesity_levels']>33))].groupby("Country").Obesity_levels.sum().sort_values(ascending = False)
Most_obese_country.plot(kind = "bar",title='Most Obese Countries in the World')
