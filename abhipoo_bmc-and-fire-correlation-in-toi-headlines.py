import pandas as pd
import seaborn as sns

df = pd.read_csv('../input/india-news-headlines.csv')

df.head()

df['headline_category'].value_counts().filter(like = 'mumbai')

#checking out mumbai-pluses

df[df['headline_category'] == 'mumbai-pluses'].head(10)

#Should've known. These sound like Times plus articles. Skipping them

#Restricting ourselves to city.mumbai

mumbai_df = df[df['headline_category'] == 'city.mumbai']

#113284 rows 

mumbai_df.head(10)

# 3/10 of the first 10 records include 'BMC' in them. Got something to drill down further

#lower case

mumbai_df['headline_text'] = mumbai_df['headline_text'].str.lower()

mumbai_df[mumbai_df['headline_text'].str.contains('bmc')]

#3184 records contain bmc. The last of these records reflect the articles for Kamala mills fire.
#Let's see the time series graph for BMC vs fire in mumbai

#cleaning date

mumbai_df['publish_date'] = pd.to_datetime(mumbai_df['publish_date'], format='%Y%m%d')

mumbai_df['publish_date'].value_counts().plot.line()

sns.kdeplot(mumbai_df['publish_date'].value_counts())

#mumbai_df['publish_date'].value_counts()

#mumbai_df[mumbai_df['publish_date'] == '2008-10-15'].head(30)

#time-series plot

#all articles published for category 'city.mumbai' - Output same as above graph

grp_by = mumbai_df.groupby(['publish_date'])['headline_text'].count()

ts = pd.Series(grp_by.values, index = grp_by.index)

#ts.plot()

#all articles published for bmc in mumbai dataframe

bmc_df = mumbai_df[mumbai_df['headline_text'].str.contains('bmc')]

grp_by_bmc = bmc_df.groupby('publish_date')['headline_text'].count()

#grp_by_bmc.head()

ts_bmc = pd.Series(grp_by_bmc.values, index = grp_by_bmc.index)

#ts_bmc.plot()

#ts filtering based on year

import pandas as pd
import seaborn as sns

df = pd.read_csv('../input/india-news-headlines.csv')

mumbai_df = df[df['headline_category'] == 'city.mumbai']

mumbai_df['publish_date'] = pd.to_datetime(mumbai_df['publish_date'], format='%Y%m%d')

mumbai_df['headline_text'] = mumbai_df['headline_text'].str.lower()

grp_by = mumbai_df.groupby(['publish_date'])['headline_text'].count()

ts = pd.Series(grp_by.values, index = grp_by.index)

#ts['2017':].plot()

#sns.kdeplot(ts['2017':])

bmc_df = mumbai_df[mumbai_df['headline_text'].str.contains('bmc')]

grp_by_bmc = bmc_df.groupby('publish_date')['headline_text'].count()

ts_bmc = pd.Series(grp_by_bmc.values, index = grp_by_bmc.index)

#ts_bmc['2017':].plot()

#sns.kdeplot(ts_bmc['2017':])

bmcfire_df = mumbai_df[(mumbai_df['headline_text'].str.contains('bmc')) & (mumbai_df['headline_text'].str.contains('fire')) ]

#bmcfire_df.head()

grp_by_bmc = bmc_df.groupby('publish_date')['headline_text'].count()

#grp_by_bmc.shape

ts_bmcfire = pd.Series(grp_by_bmc.values, index = grp_by_bmc.index)

ts_bmcfire['2017' : ].plot()

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