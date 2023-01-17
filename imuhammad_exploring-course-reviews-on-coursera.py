# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
reviews = pd.read_csv('../input/course-reviews-on-coursera/Coursera_reviews.csv')

courses = pd.read_csv('../input/course-reviews-on-coursera/Coursera_courses.csv')



merged_reviews_courses = pd.merge(reviews,courses,on = 'course_id')

merged_reviews_courses.head()
print(merged_reviews_courses.shape)

top_reviewed_courses = merged_reviews_courses.name.value_counts()

top_reviewed_courses.head(10)
top_reviewed_courses.index
sns.barplot(x = top_reviewed_courses.head(10).values ,y =top_reviewed_courses.head(10).index)
sns.barplot(x = top_reviewed_courses.tail(10).values ,y =top_reviewed_courses.tail(10).index)
top_reviewed_institution = merged_reviews_courses.institution.value_counts()

top_reviewed_institution.head(10)
sns.barplot(x = top_reviewed_institution.head(10).values ,y =top_reviewed_institution.head(10).index)
sns.barplot(x = top_reviewed_institution.tail(10).values ,y =top_reviewed_institution.tail(10).index)
merged_reviews_courses.groupby(['institution'])['rating'].mean().sort_values(ascending  = False)