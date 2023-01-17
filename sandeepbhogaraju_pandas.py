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

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']

print(heights_A.shape)



weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']

print(weights_A.dtype)



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A

print(df_A.shape)





my_mean = 170.0

my_std = 25

np.random.seed(100)

heights_B =  pd.Series(np.random.normal(loc=my_mean, scale=my_std, size=5))

heights_B.index = ['s1', 's2', 's3', 's4','s5']





my_mean1 = 75.0

my_std1 = 12

weights_B  = pd.Series(np.random.normal(loc=my_mean, scale=my_std, size=5))

weights_B.index = ['s1', 's2', 's3', 's4','s5']

print(heights_B.mean())



df_B = pd.DataFrame()

df_B['Student_height'] = heights_B

df_B['Student_weight'] = weights_B

print(df_B.shape)





p = pd.Panel()

data = {'ClassA' : df_A, 

   'ClassB' : df_B}

print(p)
import pandas as pd

import numpy as np



DatetimeIndex  = pd.date_range(start='09/1/2017', end='09/15/2017')

print(DatetimeIndex[2])



datelist = ['14-Sep-2017', '9-Sep-2017']

dates_to_be_searched = pd.to_datetime(datelist)

print(dates_to_be_searched)



print(dates_to_be_searched.isin(DatetimeIndex))



arraylist = [['classA']*5 + ['classB']*5, ['s1', 's2', 's3','s4', 's5']*2]

mi_index  = pd.MultiIndex.from_product(arraylist, names=['First Level','Second Level'])

print(mi_index.levels)
import pandas as pd

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']





weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A



df_A.loc['s3'] = np.nan

df_A.loc['s5'][1] = np.nan



df_A2 = df_A.dropna(how ='any')

print(df_A2)
import pandas as pd

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']





weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A



df_A.to_csv('classA.csv')





df_A2 = pd.read_csv('classA.csv')

print(df_A2)



df_A3 = pd.read_csv('classA.csv',index_col='Unnamed: 0')

print(df_A3)





my_mean = 170.0

my_std = 25.0

np.random.seed(100)

heights_B =  pd.Series(np.random.normal(loc=my_mean, scale=my_std, size=5))

heights_B.index = ['s1', 's2', 's3', 's4','s5']



my_mean1 = 75.0

my_std1 = 12.0

np.random.seed(100)

weights_B  = pd.Series(np.random.normal(loc=my_mean1, scale=my_std1, size=5))

weights_B.index = ['s1', 's2', 's3', 's4','s5']



df_B = pd.DataFrame()

df_B['Student_height'] = heights_B

df_B['Student_weight'] = weights_B



df_B.to_csv('classB.csv', index=False)





df_B2 = pd.read_csv('classB.csv')

print(df_B2)



df_B3 = pd.read_csv('classB.csv',header=None)

print(df_B3)



df_B4 = pd.read_csv('classB.csv',header=None,skiprows=2)

print(df_B4)
import pandas as pd

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']





weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A



df_A['Gender'] = ['M', 'F', 'M', 'M', 'F']



s = pd.Series([165.4, 82.7, 'F'],index=['Student_height', 'Student_weight', 'Gender'],name='s6')



df_AA = df_A.append(s)

print(df_AA)



my_mean = 170.0

my_std = 25.0

np.random.seed(100)

heights_B =  pd.Series(np.random.normal(loc=my_mean, scale=my_std, size=5))

heights_B.index = ['s1', 's2', 's3', 's4','s5']



my_mean1 = 75.0

my_std1 = 12.0

np.random.seed(100)

weights_B  = pd.Series(np.random.normal(loc=my_mean1, scale=my_std1, size=5))

weights_B.index = ['s1', 's2', 's3', 's4','s5']



df_B = pd.DataFrame()

df_B['Student_height'] = heights_B

df_B['Student_weight'] = weights_B



df_B.index = [ 's7', 's8', 's9', 's10', 's11']

df_B['Gender'] = ['F', 'M', 'F', 'F', 'M']



df = pd.concat([df_AA,df_B])

print(df)
import pandas as pd

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']

print(heights_A[1])

print(heights_A[[1,2,3]])



weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A



height = df_A['Student_height']

print(type(height))



df_s1s2 = df_A[df_A.index.isin(['s1','s2'])]

print(df_s1s2)



df_s2s5s1 = df_A[df_A.index.isin(['s1','s2','s5'])]

df_s2s5s1 = df_s2s5s1.reindex(['s2', 's5', 's1'])

print(df_s2s5s1)



df_s1s4 = df_A[df_A.index.isin(['s1','s4'])]

print(df_s1s4)
import pandas as pd

import numpy as np



heights_A =  pd.Series([176.2, 158.4, 167.6, 156.2, 161.4])

heights_A.index = ['s1', 's2', 's3', 's4','s5']

print(heights_A[1])

print(heights_A[[1,2,3]])



weights_A = pd.Series([85.1, 90.2, 76.8, 80.4 , 78.9])

weights_A.index = ['s1', 's2', 's3', 's4','s5']



df_A = pd.DataFrame()

df_A['Student_height'] = heights_A

df_A['Student_weight'] = weights_A



df_A_filter1 = df_A[(df_A.Student_height > 160.0) & (df_A.Student_weight < 80.0)]

print(df_A_filter1)



df_A_filter2 = df_A[df_A.index.isin(['s5'])]

print(df_A_filter2)



df_A['Gender'] =  ['M', 'F', 'M', 'M', 'F']

df_groups = df_A.groupby('Gender')

print(df_groups.mean())



nameid = pd.Series(range(101, 111))

name = pd.Series(['person' + str(i) for i in range(1, 11)])

master = pd.DataFrame()

master['nameid'] = nameid

master['name'] = name



transaction = pd.DataFrame({'nameid':[108, 108, 108,103], 'product':['iPhone', 'Nokia', 'Micromax', 'Vivo']})



mdf = pd.merge(master,transaction,on='nameid')

print(mdf)