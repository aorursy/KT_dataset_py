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
import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import warnings

import seaborn as sns

warnings.filterwarnings('ignore') 
df=pd.read_csv('../input/wine-quality/winequalityN.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df.columns
df.select_dtypes(exclude=['int','float']).columns
print(df['type'].unique())
df.info()
df.describe().T
df.hist(column='alcohol',bins=15,grid=False,figsize=(10,6),color='r')

plt.ioff()
sns.distplot(df['alcohol'],bins=25,kde=False,color='r')

plt.ioff()
sns.distplot(df['alcohol'],bins=25,kde=True,color='r')

plt.ioff()
df['alcohol'].value_counts().head()
import matplotlib.pyplot as plt

sns.distplot(df.alcohol)

plt.xlabel('Alcohol Percentage')

plt.ylabel('Count')

plt.title('Alcohol Content')

plt.ioff()
sns.set_style('dark')

sns.distplot(df.alcohol,bins=15)

plt.ioff()
df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates','alcohol','quality']].hist(figsize=(10,8),bins=40,color='r',linewidth='1.5',edgecolor='k')

plt.tight_layout()

plt.show()
sns.lmplot(x='alcohol',y='fixed acidity',data=df)

plt.ioff()
sns.lmplot(x='alcohol',y='density',data=df,fit_reg=False)

plt.ioff()
sns.lmplot(x='alcohol',y='chlorides',data=df,fit_reg=False,hue='quality')

plt.ioff()
sns.lmplot(x='alcohol',y='chlorides',data=df,fit_reg=False,hue='type')

plt.ioff()
print(df['alcohol'].quantile(0.1))

print(df['alcohol'].quantile(0.5))

print(df['alcohol'].quantile(0.9))

print(df['alcohol'].quantile(0.99))
df['alcohol'].max()
df['alcohol'].quantile(([0.05,0.95]))
import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline

import numpy as np

img=np.array(Image.open('../input/box-plot-1/BOX_PLOT.PNG'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
print(df['alcohol'].quantile(([0.25,0.75])))

sns.boxplot(data=df['alcohol'])

plt.ioff()
plt.figure(figsize=(20,10))

sns.boxplot(data=df,palette='Set3')

plt.ioff()
sns.violinplot(data=df['alcohol'])

plt.ioff()
plt.figure(figsize=(12,8))

sns.countplot(x='quality',data=df,hue='type')

plt.ioff()
sns.kdeplot(df.alcohol,df.density)

plt.ioff()
sns.jointplot(x='alcohol',y='density',data=df)

plt.ioff()
#g=sns.factorplot(x='free sulfur dioxide',y='alcohol',data=df,col='quality',hue='quality',kind='point')

#g.set_xticklabels(rotation=-45)

#plt.ioff()
import numpy

x=np.sort(df['alcohol'])

y=np.arange(1,len(x)+1)/len(x)

plt.plot(x,y,marker='.',linestyle='none')

plt.margins(0.05)

plt.xlabel('Percent of Alcohol in Wine')

plt.ylabel('ECDF')

plt.grid(True)

plt.show()
print(df['alcohol'].quantile(([0.2,0.8])))