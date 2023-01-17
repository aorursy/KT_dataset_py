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
# Import modules

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')

df.head()
df.columns
# No null values in dataset

# Drop unwanted column

df.drop('Unnamed: 0' , axis = 1 , inplace = True)
df['course_students_enrolled']=df['course_students_enrolled'].str.replace('k', '*1000')

df['course_students_enrolled']=df['course_students_enrolled'].str.replace('m', '*1000000')

df['course_students_enrolled'] = df['course_students_enrolled'].map(lambda x: eval(x))

df.info(())
df['course_students_enrolled'].describe()
# 75% courses have students a below 995000

sns.set_style('darkgrid')

sns.set_context('notebook')

bins = np.arange(1000 , 995000 , 100000 )

plt.hist(df['course_students_enrolled'] , bins = bins)

plt.xlabel('Students Count')

plt.title('Distribution of Students')
# Explore the cluster values

bins = np.arange(1000 , 20100 , 1000)

plt.hist(df['course_students_enrolled'] , bins = bins)

plt.xlabel('Students Count')

plt.title('Distribution of Students')
# Explore ratings

df['course_rating'].describe()
bins = np.arange(4 , 5.1 , 0.1)

plt.hist(df['course_rating'] , bins = bins)

plt.xlabel('Course rating')

plt.title('Distribution of course ratings')

plt.figure(figsize = (10,7))

sns.countplot(data = df , x ='course_Certificate_type' , hue = 'course_difficulty' )
plt.figure(figsize = (10,6))



plt.subplot(1,3,1)

df_data = df.groupby(['course_rating' , 'course_difficulty']).size().reset_index()

df_new = df_data.pivot('course_rating' , 'course_difficulty', 0)

sns.heatmap(df_new , cmap='rocket_r')



plt.subplot(1,3,3)

df_data = df.groupby(['course_rating'  , 'course_Certificate_type' ]).size().reset_index()

df_new = df_data.pivot('course_rating'  , 'course_Certificate_type' ,  0)

sns.heatmap(df_new , cmap='rocket_r')