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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bq_helper import BigQueryHelper 
bq_assistant = BigQueryHelper("bigquery-public-data", "github-repos")
query = """


select date as date, 
state_name as state, 
sum(confirmed_cases) as total_confirmed_cases,
sum(deaths) as total_deaths

from `bigquery-public-data.covid19_nyt.us_states`

group by 1, 2
order by 1 desc

"""

bq_assistant.estimate_query_size(query)
dataset = bq_assistant.query_to_pandas_safe(query)
dataset.head()
dataset.plot(style = 'k.')
plt.show()
dataset_daily = dataset.groupby("date")['total_confirmed_cases'].sum()
dataset_daily
dataset_daily2 = dataset.groupby("date")['total_confirmed_cases'].agg(total_cases=('total_confirmed_cases', 'sum'))
dataset_daily2
sns.lmplot(x="date", y="total_cases", data=dataset_daily2, legend = True)
plt.show()
dataset_daily2.columns
dataset_daily3 = dataset_daily2.reset_index()
sns.set_style('whitegrid')
sns.set(font_scale=1.1)

sns.lineplot(x='date', y='total_cases', data= dataset_daily3)
plt.ylabel("total_cases_in_millions")
plt.xlabel("months+beginning")
plt.title("covid cases")

plt.xticks(rotation=45)
plt.show()
