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
df = pd.read_csv('EdX_2013 Academic Year Courses.csv', encoding='iso-8859-1')
df.shape
print(df.columns)
popular = df[['Course Long Title','registered']]
popular.groupby(['Course Long Title']).sum().sort_values(by=['registered'], ascending=False)
# show most register course
popularByAge = df[['Course Long Title','registered','Age']]
popularByAge.groupby(['Age','Course Long Title']).count().sort_values(by=['registered'], ascending=False)
# show most register course by age
popularByCountry = df[['Course Long Title','registered','Country']]
popularByCountry.groupby(['Country','Course Long Title']).count().sort_values(by=['registered'], ascending=False)
# show most register course by country
popularByGender = df[['Course Long Title','registered','gender']].fillna('NaN')
popularByGender.groupby(['Course Long Title','gender']).count().sort_values(by=['registered'], ascending=False)
# show most register course by gender (Replace miss value with NaN)
mostIncomplete = df[['Course Long Title','registered','incomplete_flag']]
mostIncomplete.groupby(['Course Long Title','incomplete_flag']).count().sort_values(by=['registered'], ascending=False)
# show most incompleted course
maxCourseTook = df[['userid_DI','registered']]
maxCourseTook.groupby(['userid_DI']).count().sort_values(by=['registered'], ascending=False)
#show maximum number of course the student registered