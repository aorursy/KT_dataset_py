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
df = pd.read_csv('../input/guns.csv')
df.info()

years = df['year'].unique()

years
groupby_year = df.groupby('year')['age'].count()

groupby_year
groupby_month = df.groupby(['year','month'])['age'].count()

groupby_month_unstack = groupby_month.unstack()

groupby_month_unstack 
import matplotlib.pyplot as plt



df_t = groupby_month_unstack.transpose()

df_t.head()
df.ix(1)
plt.plot(df_t)

plt.axis()

plt.show()
df.groupby(['year','month','place'])['age'].count().unstack() 
 

    