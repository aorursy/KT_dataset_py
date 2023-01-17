import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



from subprocess import check_output

print("# Files in this dataset:")

print(check_output(["ls", "../input"]).decode("utf8"))
# Load data

df = pd.read_csv('../input/_LeagueofLegends.csv')

# df.head()
# print("# Columns:")

# print(df.dtypes)
print("Number of matches in dataset: ", len(df))



print("\n# Matches per region")

print(df['League'].value_counts())



print("\n# Matches per year")

print(df['Year'].value_counts())
# Create final gold count column

df['goldblue_final'] = pd.Series(df['goldblue'].apply(lambda x: int(x[1:-1].split(", ")[-1])))

df['goldred_final'] = pd.Series(df['goldred'].apply(lambda x: int(x[1:-1].split(", ")[-1])))

df['golddiff_final'] = pd.Series(df['golddiff'].apply(lambda x: int(x[1:-1].split(", ")[-1])))



# Gold per minute

df['bGoldpm'] = df['goldblue_final'] / df['gamelength']

df['rGoldpm'] = df['goldred_final'] / df['gamelength']
# Plot distribution of gold differences per region



plt.figure(figsize=(20,10))

plt.title("Distributions of game gold differences per region", fontsize=15)

sns.violinplot(x='League', y='golddiff_final', data=df)

plt.show()
# More precise statistics

print("# Mean gold difference (between blue and red teams):\n")



na_rwin_mean_golddiff = df[df['League'] == 'North_America'][df['rResult'] == 1]['golddiff_final'].mean()

na_bwin_mean_golddiff = df[df['League'] == 'North_America'][df['rResult'] == 0]['golddiff_final'].mean()



print("NA: ")

print("Blue win: ", na_bwin_mean_golddiff)

print("Red win: ", na_rwin_mean_golddiff)



lck_rwin_mean_golddiff = df[df['League'] == 'LCK'][df['rResult'] == 1]['golddiff_final'].mean()

lck_bwin_mean_golddiff = df[df['League'] == 'LCK'][df['rResult'] == 0]['golddiff_final'].mean()



print("\nLCK: ")

print("Blue win: ", lck_bwin_mean_golddiff)

print("Red win: ", lck_rwin_mean_golddiff)
# Calculate gold gained by winning team

df['gold_win'] = df['goldblue_final']*df['bResult'] + df['goldred_final']*df['rResult']



# Plot distribution of gold gained by winning team per region

plt.figure(figsize=(20,10))

plt.title("Distributions of winning team's gold per region", fontsize=15)

sns.violinplot(x='League', y='gold_win', data=df)

plt.show()
# Calculate gold per minute

df['goldpm_win'] = df['bGoldpm']*df['bResult'] + df['rGoldpm']*df['rResult']



# Plot distribution of winning team's gold per minute per region

plt.figure(figsize=(20,10))

plt.title("Distributions of winning team's gold per minute", fontsize=15)

sns.violinplot(x='League', y='goldpm_win', data=df)

plt.show()
# Extract final ADC gold amount per team per game

df['goldblueADC_final'] = pd.Series(df['goldblueADC'].apply(lambda x: int(x[1:-1].split(", ")[-1])))

df['goldredADC_final'] = pd.Series(df['goldredADC'].apply(lambda x: int(x[1:-1].split(", ")[-1])))



# Calculate ADC's gold share 

df['bADCgoldshare'] = df['goldblueADC_final'] / df['goldblue_final']

df['rADCgoldshare'] = df['goldredADC_final'] / df['goldred_final']

df['meanADCgoldshare'] = 0.5*(df['bADCgoldshare'] + df['rADCgoldshare'])
# Plot ADC's gold share per region

plt.figure(figsize=(20,10))

plt.title("Proportion of gold given to ADC in 2017", fontsize=15)

sns.violinplot(x='League', y='meanADCgoldshare', data=df[df['Year'] == 2017])

plt.show()
# Plot ADC's gold share per year

plt.figure(figsize=(20,10))

plt.title("Proportion of gold given to ADCs per year", fontsize=15)

sns.violinplot(x='Year', y='meanADCgoldshare', data=df)

plt.show()
# Bazinga! Everyone's here!

role_list = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']

region_list = ['LCK', 'North_America', 'Europe', 'LMS', 'CBLOL', 'Season_World_Championship', 'Mid-Season_Invitational']

for role in role_list:

    # Extract final [role] gold amount per team per game

    df['goldblue' + role + '_final'] = pd.Series(df['goldblue' + role].apply(lambda x: int(x[1:-1].split(", ")[-1])))

    df['goldred' + role + '_final'] = pd.Series(df['goldred' + role].apply(lambda x: int(x[1:-1].split(", ")[-1])))

    

    # Calculate [role]'s gold share 

    df['b' + role + 'goldshare'] = df['goldblue' + role + '_final'] / df['goldblue_final']

    df['r' + role + 'goldshare'] = df['goldred' + role + '_final'] / df['goldred_final']

    df['mean' + role + 'goldshare'] = 0.5*(df['b' + role + 'goldshare'] + df['r' + role + 'goldshare'])

    df['meanmean' + role + 'goldshare'] = None

    

    for year in range(2015, 2018):

        for region in region_list:

            spec_mean = df[(df['Year'] == year) & (df['League'] == region)]['mean' + role + 'goldshare'].mean()

            if np.isnan(spec_mean):

                spec_mean = None

            df.loc[(df['Year'] == year) & (df['League'] == region), 'meanmean' + role + 'goldshare'] = spec_mean

        # ymean may not be necessary

        ymean = df[df['Year'] == year]['mean' + role + 'goldshare'].mean()

        df.loc[(df['Year'] == year), 'ymeanmean' + role + 'goldshare'] = ymean
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

# sns.despine(left=True)



sns.barplot(x="Year", y='meanmeanTopgoldshare', data=df, ax=axes[0, 0])

sns.barplot(x="Year", y='meanmeanJunglegoldshare', data=df, ax=axes[0, 1])

sns.barplot(x="Year", y='meanmeanMiddlegoldshare', data=df, ax=axes[1, 0])

sns.barplot(x="Year", y='meanmeanSupportgoldshare', data=df, ax=axes[1, 1])
# Plot distribution of game length per region

plt.figure(figsize=(20,10))

plt.title("Distributions of game length per region", fontsize=15)

sns.violinplot(x='League', y='gamelength', data=df)

plt.show()
# Calculate number of kills per team per game

df['rNumKills'] = pd.Series(df['rKills'].apply(lambda x: len(x[2:-2].split("], ["))))

df['bNumKills'] = pd.Series(df['bKills'].apply(lambda x: len(x[2:-2].split("], ["))))



# Calculate total number of kills across both teams per game

df['total_kills'] = df['rNumKills'] + df['bNumKills']



# Calculate number of kills by winning team per game

df['win_kills'] = df['rNumKills']*df['rResult'] + df['bNumKills']*df['bResult']
# Plot distribution of kills per game per region

plt.figure(figsize=(20,10))

plt.title("Distributions of kills per game per region", fontsize=15)

sns.violinplot(x='League', y='total_kills', data=df)

plt.show()
# Plot distribution of kills per game per year

plt.figure(figsize=(20,10))

plt.title("Distributions of kills per game per year", fontsize=15)

sns.violinplot(x='Year', y='total_kills', data=df)

plt.show()
# Plot distribution of kills by winning team per game per region



plt.figure(figsize=(20,10))

plt.title("Distributions of kills by winning team per game per region", fontsize=15)

sns.violinplot(x='League', y='win_kills', data=df)

plt.show()
# Plot mean kills per year per region as a bar chart

# Plot distribution of kills by winning team per game per year

plt.figure(figsize=(20,10))

plt.title("Distributions of kills for winning team per year", fontsize=15)

sns.violinplot(x='Year', y='win_kills', data=df)

plt.show()
# Unnecessary? Haven't used this so far.

# Create one dataframe per region



lck = df[df['League'] == 'LCK']

na = df[df['League'] == 'North_America']

eu = df[df['League'] == 'Europe']

lms = df[df['League'] == 'LMS']

worlds = df[df['League'] == 'Season_World_Championship']

cblol = df[df['League'] == 'CBLOL']

msi = df[df['League'] == 'MSI']