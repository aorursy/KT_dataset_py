import pandas as pd

import re



ISO = 'ISO-8859-1'

UTF8 = 'utf8'

PREFIX = '/kaggle/input/fatal-police-shootings-in-the-us/'

ON = ['city', 'geographic area']

FILENAMES = [

    ('ShareRaceByCity.csv', UTF8),

    ('PercentOver25CompletedHighSchool.csv', ISO),

    ('PercentagePeopleBelowPovertyLevel.csv', ISO),

    ('MedianHouseholdIncome2015.csv', ISO),

]

LETHAL_WEAPONS = ['gun', 'knife', 'vehicle', 'machete', 'sword', 'ax', 'gun and knife', 'crossbow', 'box cutter']



def read_csv(file, enc):

    df = pd.read_csv(PREFIX + file, encoding=enc)

    df.columns = [c.lower() for c in df.columns]

    return df



dfs = [read_csv(file, enc) for (file, enc) in FILENAMES]

metrics_df = dfs[0].merge(dfs[1], on=ON).merge(dfs[2], on=ON).merge(dfs[3], on=ON)

metrics_df['city'] = metrics_df['city'].map(lambda c: re.sub(r'\s(CDP|city|town)$', '', c))

metrics_df = metrics_df.rename(columns={'geographic area': 'state'})

metrics_df['city_state'] = metrics_df['city'] + ', ' + metrics_df['state']



killings_df = read_csv('PoliceKillingsUS.csv', ISO)

killings_df['city_state'] = killings_df['city'] + ', ' + killings_df['state']

killings_df['had_gun'] = killings_df['armed'].eq('gun')

killings_df['had_lethal_weapon'] = killings_df['armed'].isin(LETHAL_WEAPONS)



state_pop_df = pd.read_csv('/kaggle/input/us-state-populations-2018/State Populations.csv')

state_abbrev_df = pd.read_csv('/kaggle/input/usa-state-name-code-and-abbreviation/data.csv')

state_df = state_pop_df.merge(state_abbrev_df, on='State')

state_df = state_df.rename(columns={'Code': 'state'})

state_df['pop_100k'] = round(state_df['2018 Population'] / (10 ** 5), 1)



data_df = killings_df.merge(metrics_df, on=['city', 'state', 'city_state'])

data_df = data_df.merge(state_df[['state', 'pop_100k']], on='state')

data_df = data_df.set_index('id')

data_df.head()
# Top 10 states sorted by police killing rate



summary_df = data_df.groupby(['state', 'pop_100k']).name.count().to_frame(name='count')

summary_df = summary_df.reset_index()

summary_df['rate'] = round(summary_df['count'] / summary_df['pop_100k'], 2)

summary_df.sort_values('rate', ascending=False)[0:10]
# Top 10 states sorted by police killing unarmed rate



summary_df = data_df[data_df.armed == 'unarmed'].groupby(['state', 'pop_100k']).name.count().to_frame(name='count')

summary_df = summary_df.reset_index()

summary_df['rate'] = round(summary_df['count'] / summary_df['pop_100k'], 2)

summary_df.sort_values('rate', ascending=False)[0:10]
# Top 10 states sorted by police killing non-gun rate



summary_df = data_df[data_df.had_gun == False].groupby(['state', 'pop_100k']).name.count().to_frame(name='count')

summary_df = summary_df.reset_index()

summary_df['rate'] = round(summary_df['count'] / summary_df['pop_100k'], 2)

summary_df.sort_values('rate', ascending=False)[0:10]
# Top 10 states sorted by police killing non-lethal-weapon rate



summary_df = data_df[data_df.had_lethal_weapon == False].groupby(['state', 'pop_100k']).name.count().to_frame(name='count')

summary_df = summary_df.reset_index()

summary_df['rate'] = round(summary_df['count'] / summary_df['pop_100k'], 2)

summary_df.sort_values('rate', ascending=False)[0:10]
# Top 10 non-lethal weapons



data_df[data_df['had_lethal_weapon'] == False].armed.value_counts()[0:10]