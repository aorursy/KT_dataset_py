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
from pathlib import Path



import requests

import numpy as np

import pandas as pd



import pandas_profiling

from pandas_profiling.utils.cache import cache_file

file_name = cache_file(

    "meteorites.csv",

    "https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD",

)

    

df = pd.read_csv(file_name)

    

# Note: Pandas does not support dates before 1880, so we ignore these for this analysis

df['year'] = pd.to_datetime(df['year'], errors='coerce')



# Example: Constant variable

df['source'] = "NASA"



# Example: Boolean variable

df['boolean'] = np.random.choice([True, False], df.shape[0])



# Example: Mixed with base types

df['mixed'] = np.random.choice([1, "A"], df.shape[0])



# Example: Highly correlated variables

df['reclat_city'] = df['reclat'] + np.random.normal(scale=5,size=(len(df)))



# Example: Duplicate observations

duplicates_to_add = pd.DataFrame(df.iloc[0:10])

duplicates_to_add[u'name'] = duplicates_to_add[u'name'] + " copy"



df = df.append(duplicates_to_add, ignore_index=True)
#inline report 





report = df.profile_report(sort='None', html={'style':{'full_width': True}}, progress_bar=False)

report



# save report to file 



profile_report = df.profile_report(html={'style': {'full_width': True}})

profile_report.to_file("/tmp/example.html")
#More analysis (Unicode) and Print existing ProfileReport object inline¶



profile_report = df.profile_report(explorative=True, html={'style': {'full_width': True}})

profile_report
#notebook widgets 



profile_report.to_widgets()