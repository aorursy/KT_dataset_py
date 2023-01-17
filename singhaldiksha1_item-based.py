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
import pandas as pd

course = pd.read_csv("../input/course/Course_Rating.csv")
import numpy as np

def myfunc(s1,s2):

    s1_c = s1-s1.mean()

    s2_c = s2-s2.mean()

    return (np.sum(s1_c*s2_c)/np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2)))



def cos(v1,v2):

    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    sumxx, sumxy, sumyy = 0, 0, 0

    for i in range(len(v1)):

        x = v1[i]; y = v2[i]

        sumxx += x*x

        sumyy += y*y

        sumxy += x*y

    return sumxy/math.sqrt(sumxx*sumyy)
myfunc(course['Chinese'],course['German'])

def get_recs(course_name , course , num):

    import numpy as np

    from scipy import spatial

    reviews = []

    for name in course.columns:

        if name == course_name:

            continue

        cor = myfunc(course[course_name],course[name])

        if np.isnan(cor):

            continue

        else:

            reviews.append((name,cor))

    reviews.sort(key=lambda tup: tup[1], reverse=True)

    return reviews[:num]
recs = get_recs('Android',course,40)

recs[:5]