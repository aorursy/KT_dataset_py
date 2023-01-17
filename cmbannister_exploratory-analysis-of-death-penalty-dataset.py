import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
data = pd.read_csv("../input/database.csv")
data.head()
print(data.shape)

print(data.dtypes)
print(data.isnull().sum())
data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].map(lambda x: x.year)

executions_by_year = data.groupby(['Year'])['Name'].count()

ax = executions_by_year.plot.bar(figsize=(8,6), title="Total Number of Executions in US by Year Since 1976")

label = plt.ylabel("Total # of Executions Since 1976")

plt.show()

print ("Peak Execution Number in 1999: %d" % executions_by_year.ix[1999])
executions_by_race = data.groupby(['Race'], as_index=False)['Name'].count()

executions_by_race.rename(columns={'Name':'Total Executed'}, inplace=True)

ax = executions_by_race.plot.bar(x='Race', y='Total Executed', title='Number of Executions Since 1976 \nBy Race of Person Executed')

label = plt.ylabel("Total # of Executions Since 1976")

plt.show()
census_2010_numbers = {'Asian':14465124,'Black':37685848,'Latino':50477594, 'Native American':2247098, 'Other':604265, 'White':196817552}

for index, row in executions_by_race.iterrows():

    executions_by_race.loc[index, 'Total Executed/Million Race Pop.'] = (float(row['Total Executed'])/census_2010_numbers[row['Race']]) * 1000000

ax2 = executions_by_race.plot.bar(x='Race', y='Total Executed/Million Race Pop.', title='Number of Executions Since 1976 \nBy Race of Person Executed Per Million Race Population')

label = plt.ylabel("Total # of Executions Since 1976 Per Million Race Pop.")

plt.show()
executions_by_state = data.groupby(['State'])['Name'].count()

ax = executions_by_state.plot.bar(x='State', figsize=(8,6), title="Total # of Executions Since 1976")

txt = plt.ylabel("Total # of Executions Since 1976")
census_data_state = {'WA': 6724540.0, 'DE': 897934.0, 'FE': np.nan, 'FL': 18801310.0, 'WY': 563626.0, 'NM': 2059179.0, 'TX': 25145561.0, 'LA': 4533372.0, 'NC': 9535483.0, 'NE': 1826341.0, 'TN': 6346105.0, 'PA': 12702379.0, 'NV': 2700551.0, 'VA': 8001024.0, 'CO': 5029196.0, 'CA': 37253956.0, 'AL': 4779736.0, 'AR': 2915918.0, 'IL': 12830632.0, 'GA': 9687653.0, 'IN': 6483802.0, 'AZ': 6392017.0, 'ID': 1567582.0, 'CT': 3574097.0, 'MD': 5773552.0, 'OK': 3751351.0, 'OH': 11536504.0, 'UT': 2763885.0, 'MO': 5988927.0, 'MT': 989415.0, 'MS': 2967297.0, 'SC': 4625364.0, 'KY': 4339367.0, 'OR': 3831074.0, 'SD': 814180.0}

df = executions_by_state.to_frame()

df_ret = pd.DataFrame()

for index, row in df.iterrows():

    population =  census_data_state[index]

    df.loc[index, 'Executions/1000000 state population'] = (row['Name']/population) * 1000000

ax = df['Executions/1000000 state population'].plot.bar(title="Executions Since 1976 Per Million State Population", figsize=(8,6))

label = plt.ylabel('Executions/1000000 state population')