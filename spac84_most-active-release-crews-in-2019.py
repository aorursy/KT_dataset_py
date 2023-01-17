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
df_nydus = pd.read_csv('/kaggle/input/german-movie-bootlegs-dataset-012005-102019/nfo_nydus_org.csv')
df_nydus.head()
del df_nydus['Unnamed: 0']
df_nydus.head()
## Get the 40 Crews with the most publishings in 2019:

df_nydus['rel_date'] = pd.to_datetime(df_nydus['rel_date'], format='%Y-%m-%dT%H:%M:%S')

year = df_nydus.rel_date.dt.to_period("Y")

df_year = df_nydus.groupby([year, df_nydus['group']]).count()

df_year
df_year[df_year.index.get_level_values('rel_date')==2019].sort_values(by=['tag_movie'], ascending=False).head(40)['tag_movie']