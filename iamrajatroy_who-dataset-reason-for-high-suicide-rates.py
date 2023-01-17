import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_csv('../input/who_suicide_statistics.csv')
df.info()
df.head()
df['country'].unique()
df['year'].unique()
# counrty with highest suicide rate from 1979 to 2016
df.groupby('country')['suicides_no'].sum().idxmax()
# suicides rates in russian federation from 1979 to 2016
rus_1979_2016 = df[df['country']=='Russian Federation'][(df['year']>=1979) & (df['year']<=2016)]
plt.figure(figsize=(14,8))
plt.title('suicides rates in russian federation from 1979 to 2016')
sns.lineplot(x=rus_1979_2016['year'], y=rus_1979_2016['suicides_no'])
rus_1990_1995 = rus_1979_2016[(rus_1979_2016['year']>=1990) & (rus_1979_2016['year']<=1995)]
plt.figure(figsize=(14,8))
plt.title('suicides rates in russian federation from 1990 to 1995')
sns.lineplot(x=rus_1990_1995['year'], y=rus_1990_1995['suicides_no'])
rus_1990_1995 = rus_1979_2016[(rus_1979_2016['year']>=1990) & (rus_1979_2016['year']<=1995)]
plt.figure(figsize=(8,7))
plt.title('gender distribution suicides rates in russian federation from 1990 to 1995')
sns.countplot(rus_1990_1995['sex'], palette='Set1')
