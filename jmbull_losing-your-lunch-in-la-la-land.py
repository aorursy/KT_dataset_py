#import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline

import re
import gc

path = '../input/'

inspections = pd.read_csv(path+'inspections.csv')
violations = pd.read_csv(path+'violations.csv')

x_size = 10
y_size = 6
inspections.head(3)
inspections.dtypes
cntIns = inspections.groupby(['activity_date']).size().reset_index(name='count')

x = pd.DataFrame(pd.to_datetime(cntIns['activity_date']).dt.date)
y = pd.DataFrame(cntIns['count'])

timePlot = pd.concat([x,y], axis=1)

cntObs = timePlot['count'].sum() # count of observations
cntDays = y.shape[0] # count of days

minDate = timePlot['activity_date'].min() # date of first observation
maxDate = timePlot['activity_date'].max() # datet of last observation

dateRange = re.split('\,', str(maxDate - minDate))
dateRange = dateRange[0]

print("\n\nThe data includes", "{:,}".format(cntObs), "inspections conducted on", cntDays, "days. The date range \nspans", dateRange, "from", minDate, "to", maxDate, ".\n")

fig = plt.figure(figsize=(x_size,y_size))
ax = fig.add_subplot(111)
ax.set(xlabel='Date', ylabel='# Inspections')
ax.plot_date(x=timePlot['activity_date'], y=timePlot['count'], marker='o')

del x, y, timePlot, fig, ax
gc.collect()
print(inspections['employee_id'].describe())

print('\nThere are', inspections['employee_id'].describe()[1], 'inspectors.\n')

# Any entries that don't conform to the EE01234567 format?
badids = 0

for e in inspections['employee_id']:
    if not re.match('^EE\d{7}$', e):
        badids += 1
        
print('Number of Employee IDs that don\'t match the EE0123456 ID format: ', badids)
# Everything matches. Just drop the "EE" part to make feature numeric

inspections['employee_id'] = inspections['employee_id'].apply(lambda x: x.split('EE', 1)[1])
print(inspections['employee_id'].head(5),'\n')
print(inspections['facility_id'].describe())

print('\nThere are', "{:,}".format(inspections['facility_id'].describe()[1]), 'facilities.\n')

# Any entries that don't conform to the EE01234567 format?
badfa = 0

for f in inspections['facility_id']:
    if not re.match('^FA\d{7}$', f):
        badfa += 1
        
print('Number of Facility IDs that don\'t match the FA0123456 ID format: ', badfa)
# Everything matches. Just drop the "FA" part to make feature numeric

inspections['facility_id'] = inspections['facility_id'].apply(lambda x: x.split('FA', 1)[1])
print(inspections['facility_id'].head(5),'\n')
print(inspections['facility_address'].describe())

print('\nThere are', "{:,}".format(inspections['facility_address'].describe()[1]), 'locations.\n')
print(inspections['facility_city'].describe())

print('\nThere are', "{:,}".format(inspections['facility_city'].describe()[1]), 'cities in the data. '
      'According to Wikipedia, there are 88 cities in LA county. There is a similar '
      'number of CDPs (Census-Designated Places), so we\'re in the right range.'
     )

print('\nThe cities are:\n\n',sorted(inspections['facility_city'].unique()))
# Fix bad data
inspections['facility_city'] = inspections['facility_city'].str.replace('Kern', 'KERN')
inspections['facility_city'] = inspections['facility_city'].str.replace('NORTHRISGE', 'NORTHRIDGE')
inspections['facility_city'] = inspections['facility_city'].str.replace('Rowland Heights', 'ROWLAND HEIGHTS')
inspections['facility_city'] = inspections['facility_city'].str.replace('WINNEKA', 'WINNETKA')
pd.crosstab(index=inspections['facility_city'], columns='count').head(10)
print(inspections['facility_state'].describe())

print('\nThere are', "{:,}".format(inspections['facility_state'].describe()[1]), 'states. That doesn\'t seem right.\n')

print('Uniqe States:\n',inspections['facility_state'].unique())
pd.crosstab(index=inspections['facility_state'], columns='count')
inspections = inspections.drop(['facility_state'], axis=1)
print(inspections['facility_zip'].describe())

print('\nThere are', "{:,}".format(inspections['facility_zip'].describe()[1]), 'zip codes. Are they formatted correctly?')
inspections['facility_zip'].head(10)
# Get rid of +4 in zip
inspections['facility_zip'] = inspections['facility_zip'].apply(lambda x: x.split('-', 1)[0] if x.find('-') > -1 else x)

inspections['facility_zip'].head(10)

# Any entries that don't conform to zip format?
badzips = 0

for z in inspections['facility_zip']:
    if not re.match('\d{5}$', z):
        badzips += 1
        
print('\nNumber of zips that don\'t match 5 digit format: ', badzips)
print(inspections['owner_id'].describe())

print('\nThere are', "{:,}".format(inspections['owner_id'].describe()[1]), 'owners.\n')

# Any entries that don't conform to the EE01234567 format?
badow = 0

for o in inspections['owner_id']:
    if not re.match('^OW\d{7}$', o):
        badow += 1
        
print('Number of Owner IDs that don\'t match the OW0123456 ID format: ', badow)
# Everything matches. Just drop the "OW" part to make feature numeric

inspections['owner_id'] = inspections['owner_id'].apply(lambda x: x.split('OW', 1)[1])
print(inspections['owner_id'].head(5),'\n')
print(inspections['owner_name'].describe(), '\n')

print(inspections['owner_name'].head(10))
print(inspections['pe_description'].describe())
sorted(inspections['pe_description'].unique())
pd.crosstab(index=inspections['pe_description'], columns='count')
inspections['program_element_pe'].unique().shape
print(inspections['program_name'].describe())

print('\nThere are', "{:,}".format(inspections['program_name'].describe()[1]), 'business names. '
      'A single name, like SUBWAY, may '
      '\nrepresent numerous locations and owners.')
print('These are the unique values for program_status:\n',inspections['program_status'].unique())

print('\nThe top value, ACTIVE, accounts \nfor',"{:.2f}".format(inspections['program_status'].describe()[3]/inspections['program_status'].describe()[0]),
      '% of the observations.'
     )
print(inspections['record_id'].describe())
print(inspections['serial_number'].describe())
print(inspections['service_code'].unique())

print(inspections['service_description'].unique())

pd.crosstab(index=inspections['service_description'], columns='count')
print(inspections['grade'].describe())

print('\nThere are',inspections['grade'].describe()[1],'values for grade. '
      'The top value, A, accounts \nfor',"{:.2f}".format(inspections['grade'].describe()[3]/inspections['grade'].describe()[0]),
      '% of the values. That\'s a big class imbalance.'
     )
print(inspections['score'].describe())
inspections.hist('score')
violations.dtypes
violations.head(3)
print(violations['points'].describe())
violations.hist('points')
pd.crosstab(index=violations['points'], columns='count')
violations[(violations['points'] == 11).idxmax():].head(1)
violations[(violations['violation_code'] == 'F023').idxmax():].head(1)
violations[(violations['points'] == 11)].head(5)
print(violations['violation_code'].describe())

print('\nThere are',violations['violation_code'].describe()[1],'values for '
      'violation code.')
pd.crosstab(index=violations['violation_code'], columns='count').head(3)
pd.crosstab(index=violations['violation_code'], columns='count').tail(3)
violations['violation_description'].head(5)
print(violations['violation_status'].describe())
violations = violations.drop(['violation_status'], axis=1)
