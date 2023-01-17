pwd
import numpy as np

import pandas as pd

df=pd.read_csv('../input/autompg-dataset/auto-mpg.csv',index_col='car name')
df.head()
import seaborn as sns
sns.boxplot('mpg',data=df)
sns.pairplot(data=df)
import matplotlib.pyplot as plt

plt.scatter(df.acceleration,df.weight)

plt.xlabel('acceleration')

plt.ylabel('weight')
sns.regplot('acceleration','weight',data=df)
sns.boxplot('cylinders',data=df)
df.describe()

df.info()
df.hist('cylinders',bins=8)
x=df['horsepower']

plt.hist(x,bins=20)

plt.show()
x=pd.DataFrame(df['horsepower'])

type(x)
x.info