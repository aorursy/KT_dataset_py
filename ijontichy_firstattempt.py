# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv',header=None)
df.head()
M,N = df.values.shape
print("Number of passengers = %d, Features = %d"%(M,N))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',context='notebook')
#cols = ['Pclass','Sex','Age','SibSp','Parch','Fare'] #Elijo algunas para empezar...
cols = ['Age','Fare'] #Elijo algunas para empezar...
cols
columnsTot = df.values[0,:]
columnsTot

df2 = pd.DataFrame(df.values[1:,:])
df2.head()
df2.columns = columnsTot
df2.head()
#help(sns.pairplot)
#df2[cols]
df3 = df2[cols]
#I will eliminate the NaNs (with the median value of the column)
from sklearn.preprocessing import Imputer
im = Imputer(missing_values='NaN', strategy='median',axis=0)
im = im.fit(df3)
imputedData = im.transform(df3.values)
imputedData
df2.Age = imputedData[:,0]
df2.Fare = imputedData[:,1]
df2.tail()
df.describe()
#cols = ['Pclass','Sex','Age','SibSp','Parch','Fare']
cols = ['Sex','Age','Fare']
sns.pairplot(df2[cols])
plt.show()

np.median(df2.Age)
#I could use a LabelEncoder to encode the categorical data
from sklearn.preprocessing import LabelEncoder

sexLE = LabelEncoder()
df2['Sex'] = sexLE.fit_transform(df2['Sex'].values)
df2.head()
cols = ['Sex','Age','Fare']
sns.pairplot(df2[cols])
plt.show()
df2.dtypes
df2['Cabin'][0]
type(df2['Cabin'][0])
type(df2['Cabin'][1]) == str
np.isnan(df2['Cabin'][0])
for cabin in df2['Cabin']:
    if(type(cabin)==float and np.isnan(cabin)):
        cabin = -1
df2.head()
