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
import pandas as pd

import numpy as np

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/Salaries.csv')
df.head()
df.info()
df.drop(['Notes', 'Agency'], axis=1, inplace=True)
cols = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits']



for col in cols:

    df[col] = pd.to_numeric(df[col], errors='coerce')

    

df.info()
df['Year'].unique()
df.describe()
df[df.TotalPayBenefits < 0]
groupby_year = df.groupby(df['Year'])

groupby_year['TotalPay'].median().plot(kind='bar')
onehundredk = df[df['BasePay'] > 100000]

onehundredk.shape
from collections import Counter

job_titles = df['JobTitle'][:-1] # deleting the last element "Not provided"



words_in_titles = []



for job_title in job_titles:

    words_in_titles += job_title.lower().split()

    

words_count = Counter(words_in_titles)
words_count.most_common(200)
import collections



job_group = {'Other' : ['clerk', 'worker', 'analyst'],

             'High rank' : ['supervisor', 'senior', 'chief', 'head', 'manager', 'iii', 'sprv', \

                              'principal', 'coordinator', '3', 'sergeant', 'iv', \

                              'chf', 'dir', 'captain', 'lieutenant', 'medical examiner', 'mayor', \

                              'district attorney','deputy']}



job_group = collections.OrderedDict(sorted(job_group.items(), key = lambda t: len(t)))
def transform_func(title):

    title = title.lower()

    for key, value in job_group.items():

        for each_value in value:

            if title.find(each_value) != -1:

                return key

    return 'Other'
df['JobGroup'] = df['JobTitle'].apply(transform_func)
df.head()
outliers= df[(df['TotalPay'] > 100000) & (df['JobGroup'] == 'Other')]



outliers.shape
outliers= df[(df['TotalPay'] < 100000) & (df['JobGroup'] == 'High ')]



outliers.shape
job_titles = outliers['JobTitle']



words_in_titles = []



for job_title in job_titles:

    words_in_titles.append(job_title.lower())

    

words_count = Counter(words_in_titles)
words_count.most_common(20)
df[df['JobTitle'].str.contains('fire', case=False)].median()
df.groupby(df['JobGroup'])['TotalPay'].median().plot(kind='bar')