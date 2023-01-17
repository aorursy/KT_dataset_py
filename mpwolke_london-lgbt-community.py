# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_excel('/kaggle/input/lgbt-community-in-london/Survey of Londoners data tables.xlsx')

df1.head()
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadslonelycsv/lonely.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'lonely.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
# Let's find the null values in data



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
col_search = 'topic1'

sns.barplot(y=df[col_search].value_counts().index,x=df[col_search].value_counts())

plt.show()

print(f'Unique values: {df[col_search].unique()}')
col_search = 'topic1'

ds = df[~(df[col_search].isin(['None','before tax','Index of Multiple','Quintile']))].copy()

ds.reset_index(drop=True,inplace=True)
col_search = 'topic2'

sns.barplot(y=ds[col_search].value_counts().index,x=ds[col_search].value_counts())

plt.show()
col_search = 'totals'

sns.barplot(y=ds[col_search].value_counts().index,x=ds[col_search].value_counts())

plt.show()
col_search = 'Often/ always'

sns.barplot(y=ds[col_search].value_counts().index,x=ds[col_search].value_counts())

plt.show()
#Salário médio a patir dos ranges disponíveis

col_search = 'totals'

ds['meanTot'] = ds[col_search].fillna('$ 0/').apply(lambda x: #get the mean in ranges

                    int( #transform all in int in the end

                    (int(str(x)[str(x).rfind(' ')+1:str(x).rfind('/')].replace('.','')) # Get max in range 

                    +

                    int(str(x)[str(x).find('$')+2:str(x).find('/')].replace('.','')) #Get min

                    )/2)) #divide by 2
plt.figure(figsize=(10,4))



ax1 = plt.subplot(131)

ax2 = plt.subplot2grid((1, 3), (0, 1),colspan=2)



plt.xticks(np.arange(0,25000,step=2500))







sns.violinplot(data=ds['meanTot'],ax=ax1)

sns.lineplot(data=ds['meanTot'].value_counts(),ax=ax2)





plt.tight_layout()



plt.show()



print(f'The mean of totals is {int(ds["meanTot"].mean())},00')