import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.listdir('../input'))
print(os.path.join('input', 'crisis-data.csv'))
data = pd.read_csv('../input/crisis-data.csv')
data.head()
data.shape
data.columns = [col.lower().replace(' ', '_') for col in data.columns]
data.dtypes
data.describe()
data.info()
print(data.shape[0])

## Find NULL data percentile
per = (data.isnull().sum()/data.shape[0])*100
percents = per.iloc[per.nonzero()[0]]

print(percents)

from matplotlib import pyplot as plt
percents.plot.barh()
plt.show()
## Delete NULL's from data
data = data[data['beat'].notnull()]
data = data[data['sector'].notnull()]
data = data[data['precinct'].notnull()]
data = data[data['officer_squad_desc'].notnull()]
data = data[data['officer_precinct_desc'].notnull()]
data = data[data['officer_bureau_desc'].notnull()]
data = data[data['officer_years_of_experience'].notnull()]
data = data[data['officer_year_of_birth'].notnull()]
data = data[data['disposition'].notnull()]
data = data[data['final_call_type'].notnull()]
data = data[data['initial_call_type'].notnull()]
data = data[data['call_type'].notnull()]
data = data[data['occurred_date_/_time'].notnull()]
data.describe()
data.info()
print(data.shape[0])

per = (data.isnull().sum()/data.shape[0])*100
percents = per.iloc[per.nonzero()[0]]

print(percents)
# Template ID - Key identifying unique Crisis Templates. This should be used to generate counts.
print(len(data['template_id']))
print(max(data['template_id']))
print(min(data['template_id']))

# Disposition - Disposition of the Crisis Template 
#               (one template per person per crisis involved contact) as reported by the officer.
print(len(data['disposition'].unique()))
print(data['disposition'].unique())

print(min(data['reported_date']))
print(max(data['reported_date']))

# Precinct - Geographic precinct area where the call was located
print(len(data['precinct'].unique()))
print(data['precinct'].unique())
data['reported_date'] = pd.to_datetime(data['reported_date'].astype(str), format='%Y-%m-%d')
data['reported_date'].isnull().count()
fig, ax = plt.subplots(figsize=(14, 10))

pltData = data[['disposition', 'template_id']].groupby(['disposition'])['template_id'].nunique()
pltData.plot(kind='barh')

plt.xlabel('Template')
plt.ylabel('Disposition')
plt.title('City of Seattle - Disposition of Crisis Template')
plt.show()
pltData = data[['precinct', 'template_id']].groupby(['precinct'])['template_id'].nunique()

pltData.plot(kind='barh')
plt.xlabel('Template ID')
plt.ylabel('Precinct')
plt.title('City of Seattle - Geographic precinct area ')
plt.show()
pltPrecinct = data.groupby(data['reported_date'].dt.year)['template_id'].count()

pltPrecinct.plot()
plt.xlabel('Reported Year')
plt.ylabel('Template Count')
plt.title('City of Seattle - Geographic precinct area ')
plt.show()
