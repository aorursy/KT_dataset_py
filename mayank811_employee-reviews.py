# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
reviews = pd.read_csv('../input/employee_reviews.csv', index_col=0)

reviews['current_employee'] = reviews['job-title'].str.contains('Current', case=False)

reviews.current_employee.value_counts()
reviews.loc[reviews.current_employee == True]['overall-ratings'].mean()
reviews.loc[reviews.current_employee == False]['overall-ratings'].mean()
reviews.loc[reviews.location == 'none'].head()
reviews = reviews[(reviews.dates!='None') & (reviews.dates!=' Jan 0, 0000') & (reviews.dates!=' Nov 0, 0000')]
reviews['dates'] = pd.to_datetime(reviews.dates, format=" %b %d, %Y")
reviews['location'] = reviews.location.map(lambda cty: 'Unknown' if cty == "none" else cty)
reviews['Country'] = reviews['location'].str.extract(r"\((.*?)\)", expand=False).fillna('US')
reviews.head()
