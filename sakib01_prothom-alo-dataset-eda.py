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
paper_df = pd.read_excel('../input/prothom-alo-newpaper-headline-from-2019-to-2017/prothom__alo.xlsx')
paper_df.head()
paper_df.shape
#sorting in descending order. to get time in order

paper_df = paper_df.sort_index(ascending = False).reset_index(drop = True)
paper_df['Date'].unique()[0]
paper_df[paper_df['Title'].str.contains('ধর্ষণ')]
paper_df['rape_news'] = paper_df['Title'].str.contains('ধর্ষণ').map({True:1, False:0})


paper_df['weekday_name'] = [d.strftime('%a') for d in paper_df['Date']]

paper_df[paper_df['rape_news'] == 1]
#for d in paper_df['Date']:

#    print(str(d.week) + " " + str(d.year))
paper_df['week number'] = [str(d.week) + " " + str(d.year) for d in paper_df['Date']]
import matplotlib.pyplot as plt



plt.plot(paper_df.groupby('weekday_name').sum())
paper_df.groupby('weekday_name').sum()
plt.figure(figsize=(20,10))

plt.plot(paper_df.groupby('week number', sort = False).sum())
plt.figure(figsize=(20,10))

plt.plot(paper_df.groupby('week number', sort = False).sum()[:20])
paper_week = paper_df.groupby('week number', sort = False).sum()

paper_week[paper_week['rape_news'] == paper_week['rape_news'].max()]
paper_df[(paper_df['week number'] == '16 2019') & (paper_df['rape_news'] == 1)]
plt.figure(figsize=(20,10))

plt.plot(paper_df.groupby('Date', sort = False).sum())