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

student = pd.read_csv("../input/student.csv")
student


subset = student.drop(['Name'], axis=1)

subset.head()



testdata = [5, 4.5]



lst_dist = []



for ind in subset.index:

    #print(subset['Aptitude'][ind], subset['Communication'][ind], subset['Class'][ind])

    dist_row = np.sqrt(np.square(testdata[0] - subset['Aptitude'][ind]) + np.square(testdata[1] - subset['Communication'][ind]))

    lst_dist.append([dist_row, subset['Class'][ind]]) 

df = pd.DataFrame(lst_dist)

df.columns = ('Distance', 'class')

df_sorted = df.sort_values('Distance')



k = 3

df_sorted_kval = df_sorted.head(k)

print(df_sorted_kval)