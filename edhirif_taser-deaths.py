# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_2015 = pd.read_csv('../input/2015.csv',encoding = "ISO-8859-1")

df_2016 = pd.read_csv('../input/2016.csv',encoding = "ISO-8859-1")



df_taser_2015 = df_2015[df_2015['classification']=='Taser']

df_taser_2016 = df_2016[df_2016['classification']=='Taser']
print('number killed by taser in 2015 = ' + str(df_taser_2015['uid'].count()))

print('number killed by taser in 2016 = ' + str(df_taser_2016['uid'].count()))
print(df_taser_2015.armed.value_counts())
print(df_taser_2016.armed.value_counts())
print(df_taser_2015.raceethnicity.value_counts())
print(df_taser_2016.raceethnicity.value_counts())
print(df_taser_2015.gender.value_counts())
print(df_taser_2016.gender.value_counts())
print(df_taser_2015.month.value_counts())
print(df_taser_2016.month.value_counts())
print(df_taser_2015.age.max())

print(df_taser_2015.age.min())
print(df_taser_2016.age.max())

print(df_taser_2016.age.min())