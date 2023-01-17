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
data = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')

# Yüklediğimiz veriyi inceleyelim

data.head()
data.info()
data["course_rating"] # Series

data[["course_rating"]] # Data Frames
data = data.rename(columns = {'Unnamed: 0' : 'id'})

# Değişip değişmediğini kontrol edelim

data.head()
# id kolonundaki sayılar düzensiz şekilde ilerlediği için  başka bir dataframe üzerinde örnek olarak index değişikliği yaptık.

data_ornek = data.set_index('id')

# Kontrol

data_ornek.head()
data.index.name = 'index_name'
data[['course_title','course_difficulty']].head()
# köşeli parentez ile

data['course_title'][1]
# loc ile 

data.loc[1,['course_title']]
data.loc[1:5,'course_title':'course_rating']
data.loc[5:1:-1,'course_title':'course_rating']
data.loc[1:10,'course_rating':]
data_boolean = data.course_rating > 4.5

data[data_boolean]
first_filter = data.course_rating>4.5

second_filter = data.course_Certificate_type == 'SPECIALIZATION'

third_filter = data.course_difficulty == 'Beginner'

data[first_filter & second_filter & third_filter].head(10)
data[first_filter & second_filter & third_filter].course_title.head(10)
def minus_rating(x):

    return x-1



data.course_rating.apply(minus_rating)
data.course_rating.apply(lambda n: n+1)
data_hierarchical_indexing = data.set_index(['course_rating','course_students_enrolled'])

data_hierarchical_indexing.head(10)
data.groupby('course_students_enrolled').course_rating.mean().head(10)