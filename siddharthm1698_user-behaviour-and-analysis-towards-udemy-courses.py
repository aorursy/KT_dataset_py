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

import seaborn as sb

import numpy as np
df = pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")
df.head()
df.drop(['course_id'],axis=1,inplace = True)
df.price.value_counts()[:10]
df.groupby(['is_paid'])['content_duration'].sum().plot(kind="barh")
df.groupby(['price'])['content_duration'].sum()[:10].plot(kind="bar")
import matplotlib. pyplot as plt



sorted_counts = df['level'].value_counts()

plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,

        counterclock = False,autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2);

plt.axis('square')



plt.scatter(data = df, x = 'content_duration', y = 'level')
sb.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)
sb.regplot(data = df, x = 'num_subscribers', y = 'num_reviews', fit_reg = False,

           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})
df['year'] = pd.DatetimeIndex(df['published_timestamp']).year
sb.distplot(df['year'])
base_color = sb.color_palette()[0]

sb.barplot(data = df, x = 'year', y = 'num_subscribers', color = base_color)



df.subject.value_counts()
sb.countplot(data = df, y = 'subject', hue = 'is_paid')
base_color = sb.color_palette()[0]

sb.violinplot(data = df, y = 'subject', x = 'year', color = base_color,

              inner = 'quartile')