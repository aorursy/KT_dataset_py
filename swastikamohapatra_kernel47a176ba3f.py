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

import numpy as np



df=pd.read_csv(r'../input/habermans-survival-data-set/haberman.csv')

print(df)
df.shape
df.columns
df.columns=['Age','year','axillary nodes','survival']
df['survival']=df['survival'].map({1:'yes', 2:'no'})
df['survival']=df['survival'].astype('category')
print(df)
df['Age'].describe()

#Age of patients
df['year'].describe()

#Year of treatment
df['axillary nodes'].describe()

#No. of axillary nodes detected
df['survival'].describe()
df['survival'].value_counts()
df.mode()
'''Observations-1

1)305 rows and 4 columns

2)Mean age of patients- 52.53 years

3)age of youngest patient-30  age of Eldest patient-83

4)Years of treatment range from 1958-69

5) Mean no. of axillary nodes detected- 4

6) max no. of axillary nodes detected-52 min no. of axillary nodes detected-0

7) 224 patients survived 5 years or longer after treatment while 81 died within 5 years

8) 25% of the patients have no axillary nodes and 75% of the patients have less than 5 axillary nodes

9)Maximum number of cases occured in the year 1958'''
import seaborn as sns
pdf1=sns.FacetGrid(df , hue='survival', size=5).map(sns.distplot, 'Age').add_legend()

plt.show
pdf2=sns.FacetGrid(df , hue='survival', size=5).map(sns.distplot, 'year').add_legend()

plt.show
pdf3=sns.FacetGrid(df , hue='survival', size=5).map(sns.distplot, 'axillary nodes').add_legend()

plt.show
sns.boxplot(x='survival',y='Age', data=df)
sns.boxplot(x='survival',y='year', data=df)
sns.boxplot(x='survival',y='axillary nodes', data=df)
'''Observation-2

1)A lot of People who survived 5 years or longer have axillary nodes between 0 and 5'''
sns.pairplot(df, hue="survival", height=5)