
import os
import folium
import pandas as pd
import numpy as np
import missingno as mn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
%matplotlib inline
warnings.filterwarnings('ignore')
data_path = os.path.join(os.path.pardir, 'input', 'Accidental_Drug_Related_Deaths__2012-2017 (1).csv')
df = pd.read_csv(data_path)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]  # standardise column names
df.head()
drug_types = ['heroin', 'cocaine', 'fentanyl', 'oxycodone', 'oxymorphone', 'etoh',
              'hydrocodone', 'benzodiazepine', 'methadone', 'amphet', 'tramad',
               'morphine_(not_heroin)', 'any_opioid']

positive_flag_types = ['Y', 'y', ' Y', '1']

df['drug_type'] = 'other'
for drug in drug_types:
    df.loc[~df[drug].isin(positive_flag_types), drug] = 0
    df.loc[df[drug].isin(positive_flag_types), drug] = 1
    df[drug] = df[drug].astype(np.int8)
    df.loc[df[drug] == 1, 'drug_type'] = drug
mn.matrix(df)
df['deathloc_latitude'] = df['deathloc'].str.extract(r'(\d+\.\d+)', expand=True).values.astype(np.float32)
df['deathloc_longitude'] = -df['deathloc'].str.split(' -').str[1].str[:-1].astype(np.float32)
# Create map around the mean position
central_position = [df['deathloc_latitude'].mean(), df['deathloc_longitude'].mean()]
locations_map = folium.Map(location=central_position, zoom_start = 9)

# Colors for the different drug types
i = 0
pal = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 
       'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

# Add markers to the map according to their drug type
for drug in df['drug_type'].unique():
    
    drug_df = df[df['drug_type'] == drug]
    
    # Not interested in adding markers for 'other'
    if drug == 'other':  
        continue
        
    for case in drug_df.index[:30]:
        folium.Marker([drug_df.loc[case, 'deathloc_latitude'], drug_df.loc[case, 'deathloc_longitude']],
                       popup=drug_df.loc[case, 'drug_type'],
                       icon=folium.Icon(color=pal[i], icon='circle', prefix='fa')
                     ).add_to(locations_map)
    i += 1
locations_map
deaths_by_drug = df[drug_types].sum().sort_values(ascending=False)
fig, ax = plt.subplots(1, 1, figsize=[7, 5])
sns.barplot(x=deaths_by_drug, y=deaths_by_drug.index)
ax.set_xlabel('Total deaths over 6 years')
ax.set_title('Accidental deaths by drug type')
plt.show()
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['year'] = df['date'].dt.year
df['year_month'] = df['date'].dt.to_period('M')
annual_deaths = df.groupby('year_month')['date'].count()

fig, ax = plt.subplots(1, 1, figsize=[10, 5])
lr = LinearRegression().fit(pd.to_numeric(annual_deaths.index).values.reshape(-1, 1), 
                            annual_deaths.values.reshape(-1, 1))
trendline = lr.predict(pd.to_numeric(annual_deaths.index).values.reshape(-1, 1))
annual_deaths.plot(ax=ax, marker='o', ls='-', alpha=.9, markersize=5, color='r', label='Monthly deaths')
ax.plot(annual_deaths.index, trendline, ls=':', color='k', label='Trendline')
ax.set_ylabel('Total deaths')
ax.set_xlabel('Time')
ax.set_title('Annual accidental drug deaths')
ax.legend()
plt.show()
time_trends_by_drug = df.groupby(by=['year'])[drug_types].sum()

#TODO ROTATE PLOT
fig, ax = plt.subplots(1, 1, figsize=[17, 5])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data=time_trends_by_drug, square=True, cmap=cmap, center=0,
            linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
plt.show()
most_frequent_drugs = deaths_by_drug[:5].index
frequent_drugs_df = df[df['drug_type'].isin(most_frequent_drugs)]
fig, ax = plt.subplots(1, 1, figsize=[11, 6])
sns.violinplot(x='drug_type', y='age', hue='sex', data=frequent_drugs_df, ax=ax, split=True)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Drug type')
ax.set_title('Distribution of deceased\'s age by drug type and gender')
plt.show()
gender_counts = df['sex'].value_counts()
race_counts = df['race'].value_counts()
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])
sns.barplot(x=gender_counts, y=gender_counts.index, ax=ax)
sns.barplot(x=race_counts, y=race_counts.index, ax=ax1)
ax.set_title('Total deaths by gender')
ax.set_xlabel('Total deaths')
ax1.set_title('Total deaths by race')
ax1.set_xlabel('Total deaths')
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 1, figsize=[8, 5])
for race in df['race'].unique():
    if sum(df['race'] == race) > 100:
        sns.distplot(df.loc[df['race'] == race, 'age'], hist=False, label=race, ax=ax)
ax.set_title('Age distributions of deceased by race')
ax.autoscale()
ax.legend()
plt.show()
deaths_by_gender = deaths_by_drug.copy()
for gender in ['Male', 'Female']:
    temp_gender_df = df.loc[df['sex'] == gender, drug_types].sum().sort_values(ascending=False)
    temp_gender_df = 100 * temp_gender_df / sum(temp_gender_df)  # Change to a percentage
    deaths_by_gender = pd.DataFrame(deaths_by_gender).join(pd.DataFrame(temp_gender_df), rsuffix=gender)
deaths_by_race = deaths_by_drug.copy()
for race in ['White', 'Hispanic, White', 'Black']:
    temp_race_df = df.loc[df['race'] == race, drug_types].sum().sort_values(ascending=False)
    temp_race_df = 100 * temp_race_df / sum(temp_race_df)  # Change to a percentage
    deaths_by_race = pd.DataFrame(deaths_by_race).join(pd.DataFrame(temp_race_df), rsuffix=race)
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(14, 5))

N = len(deaths_by_gender.index[:5])
ind = np.arange(N)
width = 0.35
ax.bar(x=ind, 
       height=deaths_by_gender.iloc[:5, 1].values, 
       width=width/2, 
       label='Male')
ax.bar(x=ind + width/2, 
       height=deaths_by_gender.iloc[:5, 2].values, 
       width=width/2, 
       label='Female')
ax.set_xticks(ind)
ax.set_xticklabels(deaths_by_gender.index[:5])
ax.set_ylabel('Proportion of all deaths (%)')
ax.set_title('Deaths by drug split by gender')
ax.legend()

ax1.bar(x=ind, 
       height=deaths_by_race.iloc[:5, 1].values, 
       width=width/2, 
       label='White')
ax1.bar(x=ind + width/2, 
       height=deaths_by_race.iloc[:5, 2].values, 
       width=width/2, 
       label='White, Hispanic')
ax1.bar(x=ind + width, 
       height=deaths_by_race.iloc[:5, 3].values, 
       width=width/2, 
       label='Black')
ax1.set_xticks(ind)
ax1.set_xticklabels(deaths_by_race.index[:5])
ax1.set_ylabel('Proportion of all deaths (%)')
ax1.set_title('Deaths by drug split by race')
ax1.legend()


plt.show()
injury_places = df['injuryplace'].value_counts()
fig, ax = plt.subplots(1, 1, figsize=[8, 10])
sns.barplot(x=injury_places[:20], y=injury_places[:20].index)
plt.show()