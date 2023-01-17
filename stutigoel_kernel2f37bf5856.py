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
data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
data.head()
data.isnull().values.any()
data.info()
data.drop('HDI for year',axis=1,inplace=True)
data.head()
data.shape
data.describe()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6));
Age_Country = pd.DataFrame(data.groupby(['age','sex'],sort=True)['suicides/100k pop'].sum()).reset_index()
plot1 = Age_Country.sort_values(by=['suicides/100k pop','age'], ascending=False)
plot1.reset_index()
g = sns.barplot(x='suicides/100k pop', y='age', data=Age_Country, hue = 'sex');
plt.xticks(rotation=90)
plt.figure(figsize=(12,6));
s=pd.DataFrame(data.groupby(['country'])['suicides_no'].mean()).reset_index().head(25)
plot2 = s.sort_values(by=['suicides_no','country'], ascending=False).head(25)
plot2.reset_index()
f=sns.barplot(x='suicides_no',y='country',data=plot2)
plt.xticks(rotation=90)
from sklearn.preprocessing import MinMaxScaler

a=data.pivot_table(['suicides/100k pop','gdp_per_capita ($)'],['year'], aggfunc='mean')
scaler=MinMaxScaler()
f=pd.DataFrame(scaler.fit_transform(a))
f.columns  = ['gdp_per_capita ($)','suicides_no']
f.plot()