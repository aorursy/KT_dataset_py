# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/h1b_kaggle.csv')
data.dropna(inplace=True)
import re

data.EMPLOYER_NAME = data.EMPLOYER_NAME.apply(lambda x: re.sub(' +',' ', x.strip()))

data.JOB_TITLE = data.JOB_TITLE.apply(lambda x: re.sub(' +',' ', x.strip()))

data.WORKSITE = data.WORKSITE.apply(lambda x: re.sub(' +',' ', x.strip()))

data.SOC_NAME = data.SOC_NAME.apply(lambda x: re.sub(' +',' ', x.strip()))
data.columns
data.head()
yearly = data.groupby('YEAR')

yearly['CASE_STATUS'].count()

yearly['CASE_STATUS'].count().plot()
years = [2011, 2012, 2013, 2014, 2015, 2016]

for year in years:

    filtered = data[data['YEAR'] == year]

    print(year)

    print('---')

    print(filtered.groupby('EMPLOYER_NAME')['CASE_STATUS'].count().sort_values(ascending=False).head(10))

    print('\n')
top10 = data.groupby('EMPLOYER_NAME')['CASE_STATUS'].count().sort_values(ascending=False).head(10).reset_index()['EMPLOYER_NAME']

top10wages = data[data['EMPLOYER_NAME'].isin(top10)]['PREVAILING_WAGE'].mean()

restwages = data[~data['EMPLOYER_NAME'].isin(top10)]['PREVAILING_WAGE'].mean()

print(top10wages, restwages)

top10employers = data[data['EMPLOYER_NAME'].isin(top10)]

rest = data[~data['EMPLOYER_NAME'].isin(top10)]
top10employers.groupby(['WORKSITE', 'JOB_TITLE'])['PREVAILING_WAGE'].mean()
rest.groupby(['EMPLOYER_NAME', 'JOB_TITLE'])['PREVAILING_WAGE'].mean()