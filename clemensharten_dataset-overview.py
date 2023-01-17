import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
operations = pd.read_csv('../input/THOR_Korean_Bombing_Operations.csv')

operations.head()
operations.info()
# First Inspection: Mission Date

#Convert to Datetime

operations['MSN_DATETIME'] = pd.to_datetime(operations.MSN_DATE)



def correct_year_value(datetime):

    if datetime.year > 2020:

        datetime = datetime.replace(year = datetime.year - 100)

    return datetime

        

operations['MSN_DATETIME'] = operations['MSN_DATETIME'].apply(correct_year_value)
operations.set_index('MSN_DATETIME', drop=False, inplace=True)

operations['MSN_DATETIME'].groupby(pd.TimeGrouper(freq='1M')).count().plot(kind='bar')

operations['TOTAL_TONS'].groupby(pd.TimeGrouper(freq='1M')).sum().plot(kind='bar')
operations['BULLETS'].groupby(pd.TimeGrouper(freq='1M')).sum().plot(kind='bar')
operations['AIRFIELD_ID'].unique()