# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/adult-census-income/adult.csv")

df.head()
df.describe()
df['sex'].value_counts()
df[df['sex'] == 'Female']['age'].mean()
a = df['native.country'].count()

b= pd.DataFrame((df['native.country'].value_counts()/a)*100)

b = b.T['Germany']

pd.DataFrame(b)
income_agg = pd.pivot_table(df, 

                           index=["income"], 

                           values=['age'], 

                           aggfunc=[np.mean, np.std])

income_agg
income = pd.crosstab(df['income'], df['education'], margins=True).drop(['All'],axis=1)

income = income.drop(['All'], axis=0)

income.plot(kind='bar', figsize=[20,10])
pd.pivot_table(df, 

                index=["race"], 

                values=['age'], 

                aggfunc=[max, min, np.mean, np.median, np.std])
status = pd.crosstab(df['marital.status'], df['income'], margins=True)
status=status.drop(['All'],axis=1)

status=status.drop(['All'],axis=0)

status
hpw = df['hours.per.week'].max()
df_1 = pd.DataFrame(df[df['hours.per.week']==99].count(), columns=['no. of people'])

df_1 = df_1[12:13]

df_1['maximum no. of hours a person work/week']=hpw

print(df_1)
new = pd.crosstab(df['income'], df['hours.per.week'], margins=True)

100*(new/new.loc['All'])
pd.crosstab(df['native.country'], df['income'], 

           values=df['hours.per.week'], aggfunc=np.mean).T