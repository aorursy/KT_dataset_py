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
df = pd.read_csv('../input/celebrity_deaths_3.csv')
df.head()


df.info()
df.describe() 


gr_by_year = df.groupby('death_year')['age'].count().reset_index()

gr_by_year.sort(ascending=False)
gr_month = df.groupby(['death_year','death_month'])['age'].count()

gr_month.sort_values(ascending=False).head(10)