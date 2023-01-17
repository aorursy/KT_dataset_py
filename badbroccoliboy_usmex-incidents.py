import pandas as pd

import matplotlib.pyplot as plt
mm = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')
mm.info()
mm.head()
mm['Region of Incident'].value_counts()
#I'll be looking at US-Mexico Border

usmx = mm[mm['Region of Incident'] == 'US-Mexico Border']
# date, region and total dead and missing is complete

usmx.info()
usmx['Total Dead and Missing'].value_counts()
#looking at top 'total dead and missing' entries

usmx.sort_values('Total Dead and Missing',ascending = False).head()
# top 15 causes of death

usmx['Cause of Death'].value_counts()[:15].plot(kind = 'barh', figsize = (9,4), color = 'maroon');
#total over all 1337 incidents

usmx['Total Dead and Missing'].sum()
#filled NaN values with zero

bymn = usmx.groupby(['Reported Year','Reported Month']).agg({'Total Dead and Missing': 'count'}).unstack(fill_value = 0)

bymn
#total dead and missing by year and month

bymn.plot(kind = 'bar', legend = False, figsize = (9,4));
byyr = usmx.groupby(['Reported Year']).agg({'Total Dead and Missing': 'count'})
#total dead and missing by year

byyr.plot(kind = 'bar');