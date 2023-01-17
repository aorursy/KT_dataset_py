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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib notebook
df = pd.read_csv('/kaggle/input/woman-murdering-in-turkey-20082020/women_who_have_been_murdered_in_turkey.csv')

df['date']=df['date'].str.strip()
df['date']=pd.to_datetime(df['date'],format='%d/%m/%Y', errors='coerce')
df = df.sort_values(by='date')
df = df[(df['date'].dt.year >= 2008) & (df['date'].dt.year < 2020)]
df['year'] = df['date'].dt.year
df.set_index('date',inplace=True)

df['age'] = df['age'].str.strip()
df['age'] = df['age'].apply(lambda x: True if x=='Resit' else ('NaN' if x=='Resit Degil' else False))
df = df.rename(columns={'age': 'adult'})

df
plt.figure(figsize=(12,8))

plt.plot(df[df['adult']==True].groupby('year')['id'].count().index,df[df['adult']==True].groupby('year')['id'].count(), color='red', marker='o', label='Reşit')
plt.plot(df[df['adult']==False].groupby('year')['id'].count().index,df[df['adult']==False].groupby('year')['id'].count(), color='blue', marker='o', label='Reşit Değil')
plt.plot(df[df['adult']=='NaN'].groupby('year')['id'].count().index,df[df['adult']=='NaN'].groupby('year')['id'].count(), color='green', marker='o', label='Bilinmiyor')

plt.xticks(df[df['adult']==False].groupby('year')['id'].count().index, df[df['adult']==False].groupby('year')['id'].count().index, rotation=45)
plt.yticks(np.arange(0, max(df[df['adult']==False].groupby('year')['id'].count())+1, 20))

plt.ylabel('Cinayete kurban giden kadın sayısı')
plt.xlabel('Yıl')

plt.legend(title='Reşitlik Durumu')
plt.grid(alpha=0.5)
plt.title('Türkiye\'deki Kadın Cinayetlerinin Yıllara ve Reşitliğe Göre Dağılımı')
plt.show()



