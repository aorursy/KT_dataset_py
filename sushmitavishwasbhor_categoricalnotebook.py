import numpy as np

import pandas as pd

df=pd.read_csv('../input/myfile1/cars.csv')
df.head()
df.pivot_table(values='(kW)', index='YEAR',columns='Make', aggfunc=np.mean) 
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)
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

from pandas.api.types import CategoricalDtype

df = pd.DataFrame(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D'],

                 index = ['excellent','excellent','excellent','good','good','good','ok','ok','ok','poor','poor'])

df.rename(columns={0:'Grades'},inplace=True)

df
df['Grades'].astype('category').head()
from pandas.api.types import CategoricalDtype



cats = ['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']

cat_type = CategoricalDtype(categories=cats, ordered=True)

df['Grades'] = df['Grades'].astype(cat_type)

print (df)
df.select_dtypes(include=['number']) > 'C'

df['Grades']>'C'
s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])



s.astype('category', CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=True))

import numpy as np

df = pd.read_csv('../input/myfile/census.csv')

df = df[df['SUMLEV']==50]

df.groupby= df.set_index('STNAME').agg({'CENSUS2010POP':np.average})
%%timeit -n 10

for state in df['STNAME'].unique():

    avg = np.average(df.where(df['STNAME']== state).dropna()['CENSUS2010POP'])

    print('Countries in state ' + state + ' have an average population of '+ str(avg))
(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({np.average,np.sum}))
df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011'].agg({np.average,np.sum})
df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011'].agg({'POPESTIMATE2010':np.average,'POPESTIMATE2011':np.sum})