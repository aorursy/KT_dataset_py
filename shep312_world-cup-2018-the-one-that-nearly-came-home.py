import os
import colorlover as cl
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgbm
import plotly.offline as py
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, classification_report, confusion_matrix, precision_recall_curve
import warnings
py.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')
%matplotlib inline
input_path = os.path.join(os.pardir, 'input', 'FIFA 2018 Statistics.csv')
df = pd.read_csv(input_path)
df.head()
print('\t# Cols = {}\n\t# Rows = {}'.format(df.shape[1], df.shape[0]))
dtype_counts = df.dtypes.value_counts()

fig, ax = plt.subplots(1, 1, figsize=[7, 4])
sns.barplot(y=dtype_counts.index.astype(str), x=dtype_counts, ax=ax, 
            palette=sns.color_palette("BuGn_r"))

for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.grid(axis='x', linestyle='--')
ax.set_xlabel('Variable count')

plt.suptitle('Distribution of data types', ha='left', fontsize=16, x=.125, y=1)
plt.title('Mostly numeric, with a handful of categoricals', ha='left', x=0)
plt.show()
null_sums = df.isnull().sum()
null_sums = null_sums[null_sums > 0].sort_values(ascending=False)

fig, ax = plt.subplots(1, 1, figsize=[7, 4])
sns.barplot(y=null_sums.index, x=100 * null_sums / len(df), 
            ax=ax, palette=sns.color_palette("Blues"))

for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.grid(axis='x', linestyle='--')
ax.set_xlabel('Null %')

plt.suptitle('Null % of columns', ha='left', fontsize=16, x=.125, y=1)
plt.title('Only columns with at least one null value plotted', ha='left', x=0)
plt.show()
df['Own goals'].fillna(0, inplace=True)
match_df = df.merge(df, left_on=['Date', 'Team'], right_on=['Date', 'Opponent'], 
                    how='inner', suffixes=[' Team', ' Opponent'])

keep = []
for i, row in match_df.iterrows():
    if i > 0:
        if (row['Team Team'] == match_df.loc[i - 1, 'Opponent Team']) & \
            (row['Date'] == match_df.loc[i - 1, 'Date']):
            continue
        else:
            keep.append(i)
            
match_df = match_df.loc[keep, :]
match_df.head()
match_df.loc[match_df['Goal Scored Team'] > match_df['Goal Scored Opponent'], 'Result'] = 'Team win'
match_df.loc[match_df['Goal Scored Team'] < match_df['Goal Scored Opponent'], 'Result'] = 'Opponent win'
match_df.loc[match_df['Goal Scored Team'] == match_df['Goal Scored Opponent'], 'Result'] = 'Draw'
match_df.loc[(match_df['Goal Scored Team'] == match_df['Goal Scored Opponent']) &
             (match_df['Goals in PSO Team'] < match_df['Goals in PSO Opponent']), 'Result'] = 'Opponent win (Pens)'
match_df.loc[(match_df['Goal Scored Team'] == match_df['Goal Scored Opponent']) &
             (match_df['Goals in PSO Team'] > match_df['Goals in PSO Opponent']), 'Result'] = 'Team win (Pens)'
results_count = match_df['Result'].value_counts()

fig, ax = plt.subplots(1, 1, figsize=[7, 4])
sns.barplot(y=results_count.index, x=results_count, ax=ax, 
            palette=sns.cubehelix_palette(5))

for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.grid(axis='x', linestyle='--')
ax.set_xlabel('Count')

plt.suptitle('Match outcomes inferred from goals scored', ha='left', fontsize=16, x=.125, y=1)
plt.title('Roughly equal wins between \'Team\' and \'Opponent\'', ha='left', x=0)
plt.show()
match_df['Distance vs. Opponent'] = match_df['Distance Covered (Kms) Team'] - match_df['Distance Covered (Kms) Opponent']
match_df['Goal Difference'] = match_df['Goal Scored Team'] - match_df['Goal Scored Opponent']

lm1 = LinearRegression().fit(match_df['Distance vs. Opponent'].values.reshape(-1, 1), match_df['Ball Possession % Team'])
extremes = np.array([match_df['Distance vs. Opponent'].min(), match_df['Distance vs. Opponent'].max()]).reshape(-1, 1)
poss_pred = lm1.predict(match_df['Ball Possession % Team'].values.reshape(-1, 1))
poss_pred_plot = lm1.predict(extremes)

lm2 = LinearRegression().fit(match_df['Distance vs. Opponent'].values.reshape(-1, 1), match_df['Goal Difference'])
gd_pred = lm2.predict(match_df['Ball Possession % Team'].values.reshape(-1, 1))
gd_pred_plot = lm2.predict(extremes)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 4])

ax.scatter(match_df['Distance vs. Opponent'], match_df['Ball Possession % Team'], edgecolors='blue', alpha=.3,
           s=100, c='blue')
ax.plot(extremes, poss_pred_plot, color='k', linestyle='--', 
        label='Linear fit (R2 = {:.2f})'.format(r2_score(match_df['Ball Possession % Team'], poss_pred)))
ax.set_ylabel('Possession (%)')
ax.set_title('On possession', ha='left', fontsize=12, x=0, y=1.05)

ax1.scatter(match_df['Distance vs. Opponent'], match_df['Goal Difference'], edgecolors='green', alpha=.3,
           s=100, c='green')
ax1.plot(extremes, gd_pred_plot, color='k', linestyle='--', 
        label='Linear fit (R2 = {:.2f})'.format(r2_score(match_df['Ball Possession % Team'], gd_pred)))
ax1.set_ylabel('Match goal difference')
ax1.set_title('On goal difference', ha='left', fontsize=12, x=0, y=1.05)

for a in (ax, ax1):
    a.legend(frameon=False)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.set_xlabel('Distance run further than opponent (km)')

plt.suptitle('Effects of running further than the opponent', ha='left', x=.125, fontsize=16, y=1.05)
plt.show()
fig, (ax, ax1) = plt.subplots(2, 1, figsize=[14, 11])

for res in match_df['Result'].unique():
    sns.kdeplot(match_df.loc[match_df['Result'] == res, 'Distance vs. Opponent'], 
                ax=ax, label=res, shade=True)
ax.set_title('Distribution of distance run vs. opponent', ha='left', fontsize=16, x=0, y=1)
ax.set_xlabel('Distance run - distance run by opponent (km)')    
for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.yaxis.set_visible(False)
ax.set_ylim([ax.get_ylim()[0], 0.13])
ax.legend(frameon=False)

order = df.groupby('Team')['Distance Covered (Kms)'].mean().sort_values().index
sns.boxplot(x='Team', y='Distance Covered (Kms)', data=df, order=order)
ax1.set_title('Average distances run by each team involved', ha='left', fontsize=16, x=0, y=1)
ax1.set_xlabel('')
ax1.set_ylabel('Distance per match (km)')
for spine in ['top', 'left', 'right', 'bottom']:
    ax1.spines[spine].set_visible(False)
ax1.grid(linestyle='--', alpha=.3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

plt.autoscale()
plt.show()
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])

for res in match_df['Result'].unique():
    sns.kdeplot(match_df.loc[match_df['Result'] == res, 'Ball Possession % Team'], 
                ax=ax, label=res, shade=True)
ax.set_title('Whilst a minor effect, it seems possession increases win probability slightly', 
             ha='left', fontsize=12, x=0, y=1.02)
ax.yaxis.set_visible(False)
ax.set_ylim([ax.get_ylim()[0], .04])
    
sns.barplot(y='Result', x='Ball Possession % Team', data=match_df, ax=ax1)
ax1.grid(axis='x', linestyle='--')
ax1.set_ylabel('')

for a in [ax, ax1]:
    for spine in ['top', 'left', 'right']:
        a.spines[spine].set_visible(False)
    a.set_xlabel('Team possession (%)')
    a.legend(frameon=False)

plt.suptitle('Possession distributions and their mean values', ha='left', x=.125, fontsize=16, y=1)
plt.autoscale()
plt.show()
country_dict = {
    'Afghanistan': 'AFG',
     'Albania': 'ALB',
     'Algeria': 'DZA',
     'American Samoa': 'ASM',
     'Andorra': 'AND',
     'Angola': 'AGO',
     'Anguilla': 'AIA',
     'Antigua and Barbuda': 'ATG',
     'Argentina': 'ARG',
     'Armenia': 'ARM',
     'Aruba': 'ABW',
     'Australia': 'AUS',
     'Austria': 'AUT',
     'Azerbaijan': 'AZE',
     'Bahamas, The': 'BHM',
     'Bahrain': 'BHR',
     'Bangladesh': 'BGD',
     'Barbados': 'BRB',
     'Belarus': 'BLR',
     'Belgium': 'BEL',
     'Belize': 'BLZ',
     'Benin': 'BEN',
     'Bermuda': 'BMU',
     'Bhutan': 'BTN',
     'Bolivia': 'BOL',
     'Bosnia and Herzegovina': 'BIH',
     'Botswana': 'BWA',
     'Brazil': 'BRA',
     'British Virgin Islands': 'VGB',
     'Brunei': 'BRN',
     'Bulgaria': 'BGR',
     'Burkina Faso': 'BFA',
     'Burma': 'MMR',
     'Burundi': 'BDI',
     'Cabo Verde': 'CPV',
     'Cambodia': 'KHM',
     'Cameroon': 'CMR',
     'Canada': 'CAN',
     'Cayman Islands': 'CYM',
     'Central African Republic': 'CAF',
     'Chad': 'TCD',
     'Chile': 'CHL',
     'China': 'CHN',
     'Colombia': 'COL',
     'Comoros': 'COM',
     'Congo, Democratic Republic of the': 'COD',
     'Congo, Republic of the': 'COG',
     'Cook Islands': 'COK',
     'Costa Rica': 'CRI',
     "Cote d'Ivoire": 'CIV',
     'Croatia': 'HRV',
     'Cuba': 'CUB',
     'Curacao': 'CUW',
     'Cyprus': 'CYP',
     'Czech Republic': 'CZE',
     'Denmark': 'DNK',
     'Djibouti': 'DJI',
     'Dominica': 'DMA',
     'Dominican Republic': 'DOM',
     'Ecuador': 'ECU',
     'Egypt': 'EGY',
     'El Salvador': 'SLV',
     'Equatorial Guinea': 'GNQ',
     'Eritrea': 'ERI',
     'Estonia': 'EST',
     'Ethiopia': 'ETH',
     'Falkland Islands (Islas Malvinas)': 'FLK',
     'Faroe Islands': 'FRO',
     'Fiji': 'FJI',
     'Finland': 'FIN',
     'France': 'FRA',
     'French Polynesia': 'PYF',
     'Gabon': 'GAB',
     'Gambia, The': 'GMB',
     'Georgia': 'GEO',
     'Germany': 'DEU',
     'Ghana': 'GHA',
     'Gibraltar': 'GIB',
     'Greece': 'GRC',
     'Greenland': 'GRL',
     'Grenada': 'GRD',
     'Guam': 'GUM',
     'Guatemala': 'GTM',
     'Guernsey': 'GGY',
     'Guinea': 'GIN',
     'Guinea-Bissau': 'GNB',
     'Guyana': 'GUY',
     'Haiti': 'HTI',
     'Honduras': 'HND',
     'Hong Kong': 'HKG',
     'Hungary': 'HUN',
     'Iceland': 'ISL',
     'India': 'IND',
     'Indonesia': 'IDN',
     'Iran': 'IRN',
     'Iraq': 'IRQ',
     'Ireland': 'IRL',
     'Isle of Man': 'IMN',
     'Israel': 'ISR',
     'Italy': 'ITA',
     'Jamaica': 'JAM',
     'Japan': 'JPN',
     'Jersey': 'JEY',
     'Jordan': 'JOR',
     'Kazakhstan': 'KAZ',
     'Kenya': 'KEN',
     'Kiribati': 'KIR',
     'Korea, North': 'PRK',
     'Korea, South': 'KOR',
     'Kosovo': 'KSV',
     'Kuwait': 'KWT',
     'Kyrgyzstan': 'KGZ',
     'Laos': 'LAO',
     'Latvia': 'LVA',
     'Lebanon': 'LBN',
     'Lesotho': 'LSO',
     'Liberia': 'LBR',
     'Libya': 'LBY',
     'Liechtenstein': 'LIE',
     'Lithuania': 'LTU',
     'Luxembourg': 'LUX',
     'Macau': 'MAC',
     'Macedonia': 'MKD',
     'Madagascar': 'MDG',
     'Malawi': 'MWI',
     'Malaysia': 'MYS',
     'Maldives': 'MDV',
     'Mali': 'MLI',
     'Malta': 'MLT',
     'Marshall Islands': 'MHL',
     'Mauritania': 'MRT',
     'Mauritius': 'MUS',
     'Mexico': 'MEX',
     'Micronesia, Federated States of': 'FSM',
     'Moldova': 'MDA',
     'Monaco': 'MCO',
     'Mongolia': 'MNG',
     'Montenegro': 'MNE',
     'Morocco': 'MAR',
     'Mozambique': 'MOZ',
     'Namibia': 'NAM',
     'Nepal': 'NPL',
     'Netherlands': 'NLD',
     'New Caledonia': 'NCL',
     'New Zealand': 'NZL',
     'Nicaragua': 'NIC',
     'Niger': 'NER',
     'Nigeria': 'NGA',
     'Niue': 'NIU',
     'Northern Mariana Islands': 'MNP',
     'Norway': 'NOR',
     'Oman': 'OMN',
     'Pakistan': 'PAK',
     'Palau': 'PLW',
     'Panama': 'PAN',
     'Papua New Guinea': 'PNG',
     'Paraguay': 'PRY',
     'Peru': 'PER',
     'Philippines': 'PHL',
     'Poland': 'POL',
     'Portugal': 'PRT',
     'Puerto Rico': 'PRI',
     'Qatar': 'QAT',
     'Romania': 'ROU',
     'Russia': 'RUS',
     'Rwanda': 'RWA',
     'Saint Kitts and Nevis': 'KNA',
     'Saint Lucia': 'LCA',
     'Saint Martin': 'MAF',
     'Saint Pierre and Miquelon': 'SPM',
     'Saint Vincent and the Grenadines': 'VCT',
     'Samoa': 'WSM',
     'San Marino': 'SMR',
     'Sao Tome and Principe': 'STP',
     'Saudi Arabia': 'SAU',
     'Senegal': 'SEN',
     'Serbia': 'SRB',
     'Seychelles': 'SYC',
     'Sierra Leone': 'SLE',
     'Singapore': 'SGP',
     'Sint Maarten': 'SXM',
     'Slovakia': 'SVK',
     'Slovenia': 'SVN',
     'Solomon Islands': 'SLB',
     'Somalia': 'SOM',
     'South Africa': 'ZAF',
     'South Sudan': 'SSD',
     'Spain': 'ESP',
     'Sri Lanka': 'LKA',
     'Sudan': 'SDN',
     'Suriname': 'SUR',
     'Swaziland': 'SWZ',
     'Sweden': 'SWE',
     'Switzerland': 'CHE',
     'Syria': 'SYR',
     'Taiwan': 'TWN',
     'Tajikistan': 'TJK',
     'Tanzania': 'TZA',
     'Thailand': 'THA',
     'Timor-Leste': 'TLS',
     'Togo': 'TGO',
     'Tonga': 'TON',
     'Trinidad and Tobago': 'TTO',
     'Tunisia': 'TUN',
     'Turkey': 'TUR',
     'Turkmenistan': 'TKM',
     'Tuvalu': 'TUV',
     'Uganda': 'UGA',
     'Ukraine': 'UKR',
     'United Arab Emirates': 'ARE',
     'United Kingdom': 'GBR',
     'United States': 'USA',
     'Uruguay': 'URY',
     'Uzbekistan': 'UZB',
     'Vanuatu': 'VUT',
     'Venezuela': 'VEN',
     'Vietnam': 'VNM',
     'Virgin Islands': 'VGB',
     'West Bank': 'WBG',
     'Yemen': 'YEM',
     'Zambia': 'ZMB',
     'Zimbabwe': 'ZWE'
}
results_to_points_home = {
    'Team win': 3,
    'Opponent win': 0,
    'Draw': 1,
    'Opponent win (Pens)': 0,
    'Tean win (Pens)': 3
}
results_to_points_away = {
    'Team win': 0,
    'Opponent win': 3,
    'Draw': 1,
    'Opponent win (Pens)': 3,
    'Tean win (Pens)': 0
}
continent_dict={
    'Russia': 'Europe',
    'Saudi Arabia': 'Asia',
    'Egypt': 'Africa',
    'Uruguay': 'South America',
    'Morocco': 'Africa',
    'Iran': 'Asia',
    'Portugal': 'Europe',
    'Spain': 'Europe',
    'France': 'Europe',
    'Australia': 'Asia',
    'Argentina': 'South America',
    'Iceland': 'Europe',
    'Peru': 'South America',
    'Denmark': 'Europe',
    'Croatia': 'Europe',
    'Nigeria': 'Africa',
    'Costa Rica': 'North & Central America',
    'Serbia': 'Europe',
    'Germany': 'Europe',
    'Mexico': 'North & Central America',
    'Brazil': 'South America',
    'Switzerland': 'Europe',
    'Sweden': 'Europe',
    'Korea Republic': 'Asia',
    'Belgium': 'Europe',
    'Panama': 'North & Central America',
    'Tunisia': 'Africa',
    'England': 'Europe',
    'Colombia': 'South America',
    'Japan': 'Asia',
    'Poland': 'Europe',
    'Senegal': 'Africa'
}

# Country code exceptions
country_dict['England'] = 'GBR'  # I'm sorry, everyone. Blame Plotly for not being anglo centric enough
country_dict['Korea Republic'] = 'KOR'
match_df['Home Team Points'] = match_df['Result'].map(results_to_points_home)
match_df['Away Team Points'] = match_df['Result'].map(results_to_points_away)

country_performance_home = match_df.groupby('Team Team')['Home Team Points'].sum().reset_index()
country_performance_away = match_df.groupby('Opponent Team')['Away Team Points'].sum().reset_index()

country_performance = country_performance_home.merge(country_performance_away, 
                                                     left_on='Team Team', right_on='Opponent Team')
country_performance['Total Points'] = country_performance['Home Team Points'] + \
    country_performance['Away Team Points']

country_performance['Team Plotly Code'] = country_performance['Team Team'].map(country_dict)
data = [ dict(
        type = 'choropleth',
        locations = country_performance['Team Plotly Code'],
        z = country_performance['Total Points'],
        text = country_performance['Team Team'],
        #autocolorscale = True,
        colorscale = 'YlOrRd',
        reversescale = True,
        colorbar = dict(
            autotick = True,
            title = 'Tournament \'points\''),
      ) ]

layout = dict(
    title = 'World Cup performance by country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Natural Earth'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
# Create a dataset grouped by continent for the Sankey
df['Team Continent'] = df['Team'].map(continent_dict)
round_counts = df.groupby(['Team Continent', 'Round'])['Team'].nunique().reset_index()

round_order = {
    'Qualified': 0,
    'Group Stage': 1, 
    'Round of 16': 2, 
    'Quarter Finals': 3, 
    'Semi- Finals': 4,
    '3rd Place': 5, 
    'Final': 6
}
round_counts['Round as Number'] = round_counts['Round'].map(round_order)
round_counts['Previous Round'] = round_counts['Round as Number'] - 1
round_counts.loc[round_counts['Round as Number'] == 6, 'Previous Round'] = 4

continent_encoder = LabelEncoder()
round_counts['Encoded Continent'] = continent_encoder.fit_transform(round_counts['Team Continent'])

continents = round_counts['Team Continent'].unique()
colors = cl.scales[str(len(continents))]['qual']['Set1']
color_dict = dict(zip(continents, colors))
round_counts['Color'] = round_counts['Team Continent'].map(color_dict)
data = dict(
    type='sankey',
    node = dict(
      pad = 10,
      thickness = 30,
      line = dict(
        color = 'black',
        width = 0.5
      ),
      label = list(round_order.keys()),
      color = 'rgb(204, 204, 204)',
    ),
    link = dict(
      source = round_counts['Previous Round'],
      target = round_counts['Round as Number'],
      value = round_counts['Team'],
      color = round_counts['Color'],
      label = round_counts['Team Continent']
  ))

layout =  dict(
    title = 'Progression by Continent',
    autosize=False,
    width=800,
    height=750,

    font = dict(
      size = 10
    )
)

fig = dict(data=[data], layout=layout)
py.iplot(fig, validate=False)
match_df['Fouls difference (team - opponent)'] = match_df['Fouls Committed Team'] - match_df['Fouls Committed Opponent']

# Exclude penalty shoot outs as they skew the data
keep = ['Pens' not in x for x in match_df['Result']]

fig, (ax, ax1) = plt.subplots(1, 2, figsize=[15, 6])
sns.violinplot(x='Result', y='Fouls difference (team - opponent)', data=match_df[keep], ax=ax, palette='Blues')
for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=.6)
ax.set_xlabel('')
ax.set_ylabel('# fouls team commited more than opponent')
ax.set_title('Fouls distribution by match outcome', ha='left', fontsize=12, x=0)

ax1.scatter(match_df['Fouls Committed Team'], match_df['Free Kicks Opponent'])
ax1.set_xlabel('Fouls commited by a team')
ax1.set_ylabel('Free kicks taken by opponent')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Correlation between fouls committed and oponent\'s free kicks', 
             ha='left', fontsize=12, x=0)

plt.suptitle('Effect of committing more fouls', ha='left', x=.125, fontsize=16)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=[14, 6])
order = df.groupby('Team')['Fouls Committed'].mean().sort_values().index
sns.boxplot(x='Team', y='Fouls Committed', data=df, order=order)
ax.set_title('Be meaner, Spain!', ha='left', fontsize=12, x=0)
ax.set_xlabel('')
ax.set_ylabel('Fouls commited per match')
for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.grid(linestyle='--', alpha=.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.suptitle('Fouls committed by team', ha='left', x=.125, fontsize=16)
plt.show()
df['Shot Accuracy %'] = 100 * df['On-Target'] / (df['On-Target'] + df['Off-Target'])
team_precision = df.groupby('Team')['Pass Accuracy %', 'Shot Accuracy %'].mean().reset_index()
team_precision = \
    team_precision.merge(country_performance[['Team Team', 'Total Points']], left_on='Team', right_on='Team Team')
fig, ax = plt.subplots(1, 1, figsize = [12, 8])
ax.scatter(team_precision['Pass Accuracy %'], team_precision['Shot Accuracy %'],
           s=100 * team_precision['Total Points'], alpha=.7)
ax.set_xlabel('Pass Accuracy (%)')
ax.set_ylabel('Shot Accuracy (%)')

for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

ax.grid(linestyle='--', alpha=.7)

for i, row in team_precision.iterrows():
    ax.annotate(row['Team'], xy=(row['Pass Accuracy %']+.3, row['Shot Accuracy %']+.5))
plt.show()
for var in ['Man of the Match Opponent', 'Man of the Match Team']:
    match_df.loc[match_df[var] == 'Yes', var] = 1
    match_df.loc[match_df[var] == 'No', var] = 0
fig, ax = plt.subplots(1, 1, figsize=[11, 5])

for mom in match_df['Man of the Match Team'].unique():
    sns.kdeplot(match_df.loc[match_df['Man of the Match Team'] == mom, 'Goal Scored Team'], 
                ax=ax, label='Awarded MotM = {}'.format(bool(mom)), shade=True)

ax.set_xlabel('Goals scored')
    
for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.yaxis.set_visible(False)

ax.legend(frameon=False)
plt.suptitle('Distributions of goals scored', ha='left', fontsize=16, x=.125, y=1)
plt.title('Centered around 0 and 1 for not MotM, around 2 for MotM', ha='left', x=0)
plt.autoscale()
plt.show()
fig, ax = plt.subplots(1, 1, figsize=[14, 5])

for mom in match_df['Man of the Match Team'].unique():
    sns.kdeplot(match_df.loc[match_df['Man of the Match Team'] == mom, 'Ball Possession % Team'], 
                ax=ax, label='Awarded MotM = {}'.format(bool(mom)), shade=True)
ax.set_title('Teams with a higher possession % were more likely to get the MotM', ha='left', fontsize=12, x=0, y=1)

ax.set_xlabel('Team possession (%)')    
for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.yaxis.set_visible(False)
ax.legend(frameon=False)

plt.suptitle('Possession distributions and getting the MotM', ha='left', x=.125, fontsize=16, y=1)
plt.autoscale()
plt.show()
match_df['Simple Round'] = match_df['Round Team']
match_df.loc[match_df['Round Team'] != 'Group Stage', 'Simple Round'] = 'Knockout Stage'

fig, ax = plt.subplots(1, 1, figsize=[9, 5])
sns.barplot(x='Result', y='Man of the Match Team', hue='Simple Round', data=match_df, palette=sns.color_palette("RdBu_r", 2))
ax.legend(frameon=False)
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
plt.suptitle('Distributions of goals scored', ha='left', fontsize=16, x=.125, y=1)
plt.title('Centered around 0 for not MotM, around 2 for MotM', ha='left', x=0)
plt.show()
losing_matches = match_df[(match_df['Man of the Match Team'] == 1) &
                          (~match_df['Result'].isin(['Team win', 'Team win (Pens)']))]
losing_matches[['Date', 'Team Team', 'Opponent Team', 'Goal Scored Team', 'Result', 'Man of the Match Team']]
df[(df['Team'] == 'Germany') & (df['Opponent'] == 'Mexico')]
# Specify the label (just in case we want to predict something else)
label_name = 'Man of the Match'

# Categorical features are the non numeric ones
categoricals = df.columns[df.dtypes == 'object'].tolist()

# Label encode them otherwise LightGBM can't use them
for cat_feat in categoricals:
    encoder = LabelEncoder()
    df[cat_feat] = encoder.fit_transform(df[cat_feat])
label = df.pop(label_name)

# Don't specify the label as a categorical
if label_name in categoricals:
    categoricals.remove(label_name)
clf = lgbm.LGBMClassifier(
    boosting_type='gbdt',
)
y_prob = cross_val_predict(
    estimator=clf, 
    cv=5, 
    X=df, 
    y=label,
    fit_params={'categorical_feature': categoricals},
    method='predict_proba'
)
y_pred = np.argmax(y_prob, axis=1)
print(classification_report(y_true=label, y_pred=y_pred))
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])

# Precision recall curve
precision, recall, _ = precision_recall_curve(label, y_prob[:, 1])
ax.step(recall, precision, color='b', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Precision - recall curve')

# Confusion matrix
cnf_matrix = confusion_matrix(label, y_pred)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(cnf_matrix, annot=True, fmt='d', ax=ax1, cmap=cmap, center=0)
ax1.set_title('Confusion matrix heatmap')
ax1.set_ylabel('True label')
ax1.set_xlabel('Predicted label')

plt.show()
clf.fit(df, label, categorical_feature=categoricals)
fig, (ax, ax1) = plt.subplots(1, 2, figsize=[11, 7])
lgbm.plot_importance(clf, ax=ax, max_num_features=20, importance_type='split')
lgbm.plot_importance(clf, ax=ax1, max_num_features=20, importance_type='gain')
ax.set_title('Importance by splits')
ax1.set_title('Importance by gain')
plt.tight_layout()
