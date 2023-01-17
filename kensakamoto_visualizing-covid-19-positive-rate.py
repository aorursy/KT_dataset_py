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
df = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_05April2020.csv')
print(df)

print(df.columns)
df['Country'] = df['Country or region'].apply(lambda x: x.split(':')[0])
v = df[['Country', 'Tests/ million']].groupby(['Country'], as_index=False).mean()
df['Negative'] = df['Tests'] - df['Positive']

print(df.columns)
v = v.sort_values(by=['Tests/ million'], ascending=False)

v.head()

v2 = df[['Country', 'Tests', 'Negative', 'Positive']].groupby(['Country'], as_index=False).sum()
v2.head()
v2['rate'] = v2['Positive'] / v2['Tests']
v2
v2 = v2.sort_values(by=['rate'], ascending=False)
ax2 = v2.plot.barh(x='Country', y='rate', fontsize=14, title='COVID-19 Positive rate', figsize=(16, 34))

ax2.invert_yaxis()
a = df[df['Country'] == "Australia" ] 
a
df