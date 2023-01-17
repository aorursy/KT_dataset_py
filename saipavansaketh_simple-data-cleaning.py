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
df=pd.read_csv('/kaggle/input/data-scientist-jobs/DataScientist.csv')

df.head(10)
df=pd.read_csv('/kaggle/input/data-scientist-jobs/DataScientist.csv')

df.drop(['index','Industry','Job Description','Competitors'],axis=1,inplace=True)

df.columns = map(str.lower, df.columns)

df.drop(df.columns[0],axis=1,inplace=True)

df['job title'] = df['job title'].apply(lambda x:x.join(x.split(',')[:1]))

df['job title'] = df['job title'].apply(lambda x:x.join(x.split('/')[:1]))

df['job title'] = df['job title'].apply(lambda x:x.join(x.split(' - ')[:1]))

df['location'] = df['location'].apply(lambda x:x.join(x.split(',')[:1]))

df['headquarters'] = df['headquarters'].apply(lambda x:x.join(x.split(',')[:1]))

df.rename(columns = {'type of ownership':'ownership','salary estimate':'salary'}, inplace = True)

df['ownership'] = df['ownership'].apply(lambda x:x.split()[-1])

df['easy apply'] = df['easy apply'].replace(['-1'],'False')

df['easy apply'] = df['easy apply'].replace(['TRUE'],'True')

df['revenue'] = df['revenue'].apply(lambda x:x.split()[-1])

df['sector'] = df['sector'].replace(['-1'],np.nan)

df['ownership'] = df['ownership'].replace(['Unknown'],np.nan)

df['size'] = df['size'].replace([-1],np.nan)

df['founded'] = df['founded'].replace([-1],np.nan)

df['rating'] = df['rating'].replace([-1],np.nan)

df['company name'] = df['company name'].apply(lambda x:x.split()[0])

df = df.dropna(how='any',axis=0)

df = df.reset_index(drop=True)

df['salary'] = df['salary'].apply(lambda x:x.split()[0])

df['founded']=df['founded'].astype(int)

df['salary']=df['salary'].apply(lambda x:x.replace('$',''))

df['salary'] = df['salary'].apply(lambda x:x.replace('K',' '))

df['size'] = df['size'].apply(lambda x:x.replace('to','-'))

df['size'] = df['size'].apply(lambda x:x.replace('+',' - 0'))

df['size']= df['size'].apply(lambda x:x.replace('employees',' '))

df['salary']= df['salary'].apply(lambda x:x.replace('(Employer',' '))

def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None

df['salary'] = df['salary'].apply(convert_sqft_to_num)

df['size']=df['size'].apply(convert_sqft_to_num)

def isrevenue(x):

    if x=="Non-Applicable":

        return 0

    elif x=="(USD)":

        return 1

df['revenue']=df['revenue'].apply(isrevenue)

df['job title'] = df['job title'].apply(lambda x:x.replace('#NAME?','-'))

df['job title'] = df['job title'].apply(lambda x:x.replace('with','-'))

print('\n','unique job titles are :',df['job title'].unique())

df.head(20)