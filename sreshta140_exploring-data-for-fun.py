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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
df
df.columns
df['course_students_enrolled']=df['course_students_enrolled'].str.replace('k', '*1000')
df['course_students_enrolled']=df['course_students_enrolled'].str.replace('m', '*1000000')
df['course_students_enrolled'] = df['course_students_enrolled'].map(lambda x: eval(x))
df['course_students_enrolled']
df.describe()
ax = df['course_Certificate_type'].value_counts().plot(kind='bar', figsize=(16,12),
                                        color="coral", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("course_Certificate_type", fontsize=18)
ax.set_ylabel("Count", fontsize=18);


# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x(), i.get_height()+.5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='dimgrey')
ax = df['course_rating'].value_counts().plot(kind='bar', figsize=(16,12),
                                        color="dodgerblue", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("course_rating", fontsize=18)
ax.set_ylabel("Count", fontsize=18);

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x(), i.get_height()+.5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='dimgrey')
ax = df['course_difficulty'].value_counts().plot(kind='bar', figsize=(16,12),
                                        color="#97ed55", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("course_difficulty", fontsize=18)
ax.set_ylabel("Count", fontsize=18);

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x(), i.get_height()+.5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='dimgrey')
ct=df.groupby(['course_title'])['course_title'].count()
ct
co=df.groupby(['course_organization'])['course_organization'].count()
co
co=pd.DataFrame(co)
co=co.rename(columns={"course_organization": "Count"})
co=co.sort_values(by=['Count'], ascending=False)
co=co.sort_values(by=['Count'], ascending=False)
co=co.reset_index()
co_top_10=co.head(10)
co_top_10
fig = plt.figure(figsize=(16, 12))
ax = fig.add_axes([0,0,1,1])
ax.bar(co_top_10['course_organization'], co_top_10['Count'])
plt.show()
ct_enrolled=df.loc[:,['course_title', 'course_students_enrolled']]
ct_enrolled=ct_enrolled.sort_values(by=['course_students_enrolled'], ascending=False)
ct_enrolled_top10=ct_enrolled.head(10)
ct_enrolled_top10
fig = plt.figure(figsize=(16, 12))
ax = fig.add_axes([0,0,1,1])
ax.bar(ct_enrolled_top10['course_title'], ct_enrolled_top10['course_students_enrolled'])
plt.show()
co_e=df.loc[:,['course_organization', 'course_students_enrolled']]
co_e=pd.DataFrame(co_e.groupby('course_organization')['course_students_enrolled'].sum())
co_e=co_e.reset_index()
co_e=co_e.sort_values(by=['course_students_enrolled'], ascending=False)
co_e_top10=co_e.head(10)
fig = plt.figure(figsize=(16, 12))
ax = fig.add_axes([0,0,1,1])
ax.bar(co_e_top10['course_organization'], co_e_top10['course_students_enrolled'])
plt.show()
df.groupby('course_Certificate_type')['course_students_enrolled'].count().plot(kind='bar', grid=True,
    figsize=(16, 12)).set_ylabel('Enrolled')
df.groupby('course_rating')['course_students_enrolled'].count().plot(kind='bar', grid=True,
    figsize=(16, 12)).set_ylabel('Enrolled')
df.groupby('course_difficulty')['course_students_enrolled'].count().plot(kind='bar', grid=True,
    figsize=(16, 12)).set_ylabel('Enrolled')
course_Certificate_type_vs_course_rating=df.groupby(['course_rating', 'course_Certificate_type'])['course_title'].count()
course_Certificate_type_vs_course_rating = course_Certificate_type_vs_course_rating.unstack().fillna(0)
ax = (course_Certificate_type_vs_course_rating).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
course_rating_vs_course_difficulty=df.groupby(['course_rating', 'course_difficulty'])['course_title'].count()
course_rating_vs_course_difficulty = course_rating_vs_course_difficulty.unstack().fillna(0)
ax = (course_rating_vs_course_difficulty).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
course_Certificate_type_vs_course_difficulty=df.groupby(['course_Certificate_type', 'course_difficulty'])['course_title'].count()
course_Certificate_type_vs_course_difficulty = course_Certificate_type_vs_course_difficulty.unstack().fillna(0)
ax = (course_Certificate_type_vs_course_difficulty).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()