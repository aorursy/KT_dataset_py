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
!wget https://public.tableau.com/s/sites/default/files/media/EdX_2013%20Academic%20Year%20Courses.csv
ll
data = pd.read_csv('EdX_2013 Academic Year Courses.csv', encoding='iso-8859-1')
data.head()
data.columns
# a

data['Course Long Title'].value_counts().reset_index(name='counts')
# b

data[['Course Long Title', 'Age']].groupby('Age')['Course Long Title'].value_counts().reset_index(name='counts').drop_duplicates('Age', keep='first')

# groupby('Age').

# .reset_index(name='counts').groupby('Age')['counts'].max()
# c

data[['Course Long Title', 'Country']].groupby('Country')['Course Long Title'].value_counts().reset_index(name='counts').drop_duplicates('Country', keep='first')
# d

data[['Course Long Title', 'gender']].groupby('gender')['Course Long Title'].value_counts().reset_index(name='counts').drop_duplicates('gender', keep='first')
# e

data[['Course Long Title', 'incomplete_flag']].groupby('Course Long Title').count().sort_values(by=['incomplete_flag'], ascending=False)
# f

data[['Course Long Title', 'userid_DI']].groupby('userid_DI')['Course Long Title'].count().max()

# .count().max()

# .value_counts().reset_index(name='counts').max()