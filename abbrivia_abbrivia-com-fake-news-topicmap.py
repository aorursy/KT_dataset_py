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




df = pd.read_csv('/kaggle/input/fake-news/fake.csv', usecols = ['title','text','language'])



df = df[df['text'].map(type) == str]

df['title'].fillna(value="", inplace=True)

df.dropna(axis=0, inplace=True, subset=['text'])

df.head()



lanugages=df['language'].unique()

print(lanugages)


dataset_name='fake_news'

for language in lanugages:

   writer = pd.ExcelWriter('Abbrivia.com-'+dataset_name+'-'+language+'.xlsx', engine='xlsxwriter')

   df[df.language == language][['title','text',]].to_excel(writer, sheet_name=('Abbrivia.com-'+dataset_name)[0:30], index = True)

   writer.save()