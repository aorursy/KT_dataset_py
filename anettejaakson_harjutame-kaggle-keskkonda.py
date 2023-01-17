import numpy as np

import pandas as pd



%matplotlib inline



data = pd.read_csv('../input/cwurData.csv')



data
data[data.country == 'Estonia']
c = pd.DataFrame(data.groupby('country').quality_of_faculty.mean())



c
c.sort_values('quality_of_faculty', ascending=False)
pd.DataFrame(data[data.year == 2015].groupby('country').size())