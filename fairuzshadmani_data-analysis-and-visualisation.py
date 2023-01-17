import pandas as pd

import numpy as np



s = pd.Series([0, 1, 4, 9, 16, 25], name = 'squares')



print(s.index)

print(s.values, s.index)

print(s[2:4])
pop2014 = pd.Series([100,99.3,95.5,93.5,92.4,84.8,84.5,78.9,74.3,72.8], index=['Java','C','C++','Python','C#','PHP',

                                                                               'JavaScript','Ruby','R','Matlab'])

pop2015 = pd.Series({'Java':100,'C':99.9,'C++':99.4,'Python':96.5,'C#':91.3,'R':84.8,'PHP':84.5,'JavaScript': 83.0,

                    'Ruby':76.2,'Matlab':72.4})



print(pop2014)

print()

print(pop2015)
print(pop2014.index)

print(pop2014.iloc[0:2])

print(pop2014.loc[:'Ruby'])
twoyears = pd.DataFrame({'2014' : pop2014,'2015' : pop2015})  

print(twoyears)
twoyears['Average'] = 0.5*(twoyears['2014'] + twoyears['2015'])

print(twoyears)
open('tips.csv','r').readlines()[:10]
tips = pd.read_csv('tips.csv')
tips.head()
tips.mean()
tips.dtypes
tips.describe()
tips.shape
tips.groupby(['gender','smoker']).mean()
pd.pivot_table(tips,'total_bill','gender','smoker')
pd.pivot_table(tips,'total_bill',['gender','smoker'], ['day','time'])
import pandas as pd

from matplotlib import pyplot as plt

url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url)



df.head()
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','flower_type']

df['flower_type']=df['flower_type'].astype('category')



df.flower_type = df.flower_type.cat.rename_categories([0,1,2])

df.head()
df['flower_type'].describe()
df.hist()

plt.show()
pd.scatter_matrix(df, diagonal='kde')

plt.show()