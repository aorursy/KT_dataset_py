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

import os

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

plt.style.use('ggplot')
df1=pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
df1.head()
df1.describe(include = "all")
df1.shape
df1.info()
# number of paid and free courses offered by udemy.



df1['is_paid'].value_counts()







                                 
df1['is_paid'].value_counts() * 100 / len(df1)
ax = df1['is_paid'].value_counts().plot(kind ='bar',

                                        figsize = (12,10),

                                            

                                    color = ('g','r'))



ax.set_title('Almost every course on udemy is paid', fontsize = 22)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Paid', fontsize = 20)

plt.grid()

plt.show()
df1['price'].min(), df1['price'].max()
bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]

df1['price'] = pd.cut(df1['price'], bins)
ax = df1[df1['is_paid'] == 1]['price'].value_counts().sort_index().plot(kind='bar', figsize = (12,8),

                                                                   width = 0.7,

                                                                   alpha = 0.7)

ax.set_title('Categorization of the courses pricewise', fontsize = 24)

ax.set_ylabel('Total number of Courses', fontsize = 20)

ax.set_xlabel('Price Range', fontsize = 20)
ax = df1['subject'].value_counts().plot(kind ='bar',

                                        figsize = (12,8),

                                        color = 'y',

                                        width = 0.6,

                                        alpha = 0.6)



ax.set_title('Most Prefered Courses Subjectwise', fontsize = 24)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Courses', fontsize = 20)

ax = df1['level'].value_counts().plot(kind ='bar', 

                                        figsize = (10,8),

                                      width = 0.8,

                                      alpha= 0.7)



ax.set_title('Courses Designed for everybody!', fontsize = 22)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Levels', fontsize = 20)

grouped = df1.groupby(['level','subject'])

grouped_pct = grouped['course_id']

grouped_pct.agg('describe')

df1.dtypes

df1.num_subscribers.describe()

df1= df1.loc[df1['num_subscribers']>114000]         

df1=df1[['course_title','num_subscribers']]

df1

plt.axis("equal")

plt.pie(df1['num_subscribers'],labels=df1['course_title'],radius=1,autopct='%.2f',explode=[0,0,0.2,0.2,0])



plt.title("Top 5 subscribed Courses on Udemy",fontsize=18,bbox={'facecolor':'.8', 'pad':1.5

                                                          },loc='center')

                                                       

plt.show()