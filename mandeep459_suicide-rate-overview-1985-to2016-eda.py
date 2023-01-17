# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.show()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()
df.info()
df.describe()
import seaborn as sns
sns.barplot(x="sex", y="suicides_no", data=df)


df["age"]=df["age"].apply(lambda x: str(x).replace('5-14 years','child') if '5-14 years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x: str(x).replace('15-24 years','youth') if '15-24 years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x: str(x).replace('25-34 years','young adult') if '25-34 years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x: str(x).replace('35-54 years','early adult') if '35-54 years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x: str(x).replace('55-74 years','adult') if '55-74 years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x: str(x).replace('75+ years','senior') if '75+ years' in str(x) else str(x))
plt.figure(figsize=(10,3))
sns.barplot(x = "age", y = "suicides_no", hue = "sex", data = df)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(16,5))
sns.barplot(df.suicides_no,df.generation)
plt.show()
sns.pairplot(data=df)
df.hist(bins=40)
alpha = 0.7
plt.figure(figsize=(10,25))
sns.countplot(y='country', data=df, alpha=alpha)
plt.title('Data country wise')
plt.show()
year_suicides = df.groupby('year')[['suicides_no']].sum().reset_index()
year_suicides.sort_values(by='suicides_no', ascending=False).style.background_gradient(cmap='Purples', subset=['suicides_no'])
plt.subplots(1,1, figsize=(10,10))
ax = sns.boxplot(x='age', y='suicides_no', hue='sex',
                 data=df)
            
ax = sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
