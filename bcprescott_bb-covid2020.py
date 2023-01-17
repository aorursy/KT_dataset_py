import numpy as np
import pandas as pd
import seaborn as sns

#List available Kaggle datasets added.
#Add more by selecting 'File -> Add or Upload Data'
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
origin = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
origin.info()
origin.head()
format_date = origin.copy()
format_date['Date'] = pd.to_datetime(format_date['Date'], format='%m/%d/%y')
format_date.info()
print('hello')
print('sad day')