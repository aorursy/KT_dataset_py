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
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv', index_col=0)

df.head()
def func(a):

    if 'k' in a:

        return float(str(a).replace('k', '')) * (10 ** 3)

    if 'm' in a:

        return float(str(a).replace('m', '')) * (10 ** 6)

    else:

        return float(a)



df['course_students_enrolled'] = df['course_students_enrolled'].apply(func)

df.head()
fig, ax = plt.subplots(figsize=(12,10))

plt.scatter(df['course_rating'], df['course_students_enrolled'], s=5)

plt.yscale('log')

plt.xlabel('course_rating')

plt.ylabel('course_students_enrolled')

plt.show()
df_sum = df.groupby('course_rating').agg({'course_students_enrolled':'sum'})

df_sum.head()
fig, ax = plt.subplots(figsize=(12, 10))

plt.plot(df_sum)

plt.xlabel('course_rating')

plt.ylabel('course_students_enrolled_sum')

plt.show()