import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10,10)
df = pd.read_csv('../input/terry-stops.csv')
df.head()
df.info()
df.isnull().any()
df = df.dropna()
df.isnull().any()
df.columns
df.columns = ['Subject_Age_Group', 'Subject_ID', 'GO_SC_Num', 'Terry_Stop_ID',
       'Stop_Resolution', 'Weapon_Type', 'Officer_ID', 'Officer_YOB',
       'Officer_Gender', 'Officer_Race', 'Subject_Perceived_Race',
       'Subject_Perceived_Gender', 'Reported_Date', 'Reported_Time',
       'Initial_Call_Type', 'Final_Call_Type', 'Call_Type', 'Officer_Squad',
       'Arrest_Flag', 'Frisk_Flag', 'Precinct', 'Sector', 'Beat']
df.columns
df[:20]
del df['Subject_ID']
del df['GO_SC_Num']
filter_age = df['Subject_Age_Group'] != '-'
df_filter_age = df[filter_age]
x = df_filter_age['Subject_Age_Group'].value_counts().index
y = df_filter_age['Subject_Age_Group'].value_counts()

fig, ax = plt.subplots()
fig.set_size_inches(15, 7)

graph_age = sns.barplot(x=x, 
            y=y, 
            order=['1 - 17', '18 - 25', '26 - 35', '36 - 45', '46 - 55', '56 and Above'] )
graph_age.set(ylabel = 'Quantity of Stopped People', 
                          xlabel = 'Age Range', 
                          title = 'Stops by Age')
plt.show()
df.Stop_Resolution.unique()
# filter_stop_resolution = df['Stop_Resolution'] != '-'
# df_filter_stop_resolution = df[filter_stop_resolution]
# x_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts().index
# y_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts()

# fig, ax = plt.subplots()
# fig.set_size_inches(15, 7)
# graph_stop_resolution = sns.barplot(x=x_df_filter_stop_resolution, y=y_df_filter_stop_resolution)
# graph_stop_resolution.set(ylabel = 'Number of Stops', 
#                           xlabel = 'Resolution Type', 
#                           title = 'Seattle Terry Stops Resolution',)
# plt.show()
# I didn't remove the string value '-' (hifen) but for this analysis, I've suppressed it with this filter
filter_stop_resolution = df['Stop_Resolution'] != '-'
# Here I'm applying our dataframe using the filter to another variable, that will be a new dataframe
df_filter_stop_resolution = df[filter_stop_resolution]
# Here you can see that I'm retrieving the indexes of the Stop_Resolution column
y_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts().index
# Here we have the values for each Stop_Resolution
x_df_filter_stop_resolution = df_filter_stop_resolution['Stop_Resolution'].value_counts()

# Now, let's create a pie chart because I think it's easier for us to understand what is happening
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
graph_stop_resolution = ax.pie(x=x_df_filter_stop_resolution, 
                               labels=y_df_filter_stop_resolution,
                               autopct='%1.1f%%')

ax.set_title('Stop Resolution Comparison')

plt.show()
len(df['Officer_ID'].unique()), len(df['Officer_ID'])
officer_counts = df['Officer_ID'].value_counts()
df_officer_counts = pd.DataFrame(officer_counts)

df_officer_counts_slice = df_officer_counts[:10]

x_counts = df_officer_counts_slice['Officer_ID'].index
y_counts = df_officer_counts_slice['Officer_ID']

fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
graph_officer_counts_ten = sns.barplot(x=x_counts, y=y_counts, data=df_officer_counts_slice, palette='winter_r')
officers_ids = officer_counts[:10].index
officers_ids
df_officer_ids_weapons = df.loc[df['Officer_ID'].isin(officers_ids)]
df_officer_ids_weapons.head()
filter_officer_ids_weapons = (df_officer_ids_weapons['Weapon_Type'] != '-') & (df_officer_ids_weapons['Weapon_Type'] != 'None')
filter_officer_ids_weapons.any()
df_officer_ids_weapons_filtered = df_officer_ids_weapons[filter_officer_ids_weapons]

sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_weapons = df_officer_ids_weapons_filtered['Weapon_Type'].value_counts()
y_weapons = df_officer_ids_weapons_filtered['Weapon_Type'].value_counts().index

graph_weapons_officers = ax.pie(x=x_weapons,
                                labels=y_weapons, 
                                autopct='%1.1f%%',
                                pctdistance=0.8)

for weapon in graph_weapons_officers[0]:
    weapon.set_edgecolor('black')
plt.show()
filter_total_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None')
filter_total_weapons.any()
df_total_weapons = df[filter_total_weapons]

# Before we go ahead, I'll fix some Weapon Types to make it easier for us.
df_total_weapons = df_total_weapons.replace({'Blackjack':'Club, Blackjack, Brass Knuckles', 
                                             'Brass Knuckle':'Club, Blackjack, Brass Knuckles',
                                             'Club':'Club, Blackjack, Brass Knuckles',
                                             'Firearm Other':'Firearm', 'Firearm (unk type)':'Firearm'})

max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(15,15)

x_total_weapons = df_total_weapons['Weapon_Type'].value_counts()[:max_weapon_on_chart]
y_total_weapons = df_total_weapons['Weapon_Type'].value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 15})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
filter_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None') 
filter_race = (df.Subject_Perceived_Race != 'Unknown') & (df.Subject_Perceived_Race != '-')

df[filter_weapons & filter_race].Subject_Perceived_Race.unique()
df_weapons_race = df[filter_weapons & filter_race]

df_weapons_race.Subject_Perceived_Race.unique()
filter_Multi_Racial = df_weapons_race.Subject_Perceived_Race == 'Multi-Racial'
filter_Black = df_weapons_race.Subject_Perceived_Race == 'Black'
filter_White = df_weapons_race.Subject_Perceived_Race == 'White'
filter_AIAN = df_weapons_race.Subject_Perceived_Race == 'American Indian / Alaskan Native'
filter_Asian = df_weapons_race.Subject_Perceived_Race == 'Asian'
filter_Hispanic = df_weapons_race.Subject_Perceived_Race == 'Hispanic'
filter_Other = df_weapons_race.Subject_Perceived_Race == 'Other'
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Multi_Racial].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Multi_Racial].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Black].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Black].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_White].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_White].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
filter_AIAN = df_weapons_race.Subject_Perceived_Race == 'American Indian / Alaskan Native'
filter_Asian = df_weapons_race.Subject_Perceived_Race == 'Asian'
filter_Hispanic = df_weapons_race.Subject_Perceived_Race == 'Hispanic'
filter_Other = df_weapons_race.Subject_Perceived_Race == 'Other'
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_AIAN].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_AIAN].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Asian].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Asian].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Hispanic].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Hispanic].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
max_weapon_on_chart = 5
#chart config
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_total_weapons = df_weapons_race[filter_Other].Weapon_Type.value_counts()[:max_weapon_on_chart]
y_total_weapons = df_weapons_race[filter_Other].Weapon_Type.value_counts().index[:max_weapon_on_chart]

graph_total_weapons = ax.pie(x=x_total_weapons, labels=y_total_weapons, autopct='%1.1f%%', textprops={'fontsize': 12})

for weapon in graph_total_weapons[0]:
    weapon.set_edgecolor('white')
    
plt.show()
df.Officer_Gender.unique().tolist()
sns.set_palette('Reds_r')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_officers_genders = df.Officer_Gender.value_counts()
y_officers_genders = df.Officer_Gender.value_counts().index

graph_officers_gender = ax.pie(x=x_officers_genders, labels=y_officers_genders, autopct='%1.2f%%')

plt.show()
filter_weapons = (df['Weapon_Type'] != '-') & (df['Weapon_Type'] != 'None') 
filter_female = df['Officer_Gender'] == 'F'
filter_male = df['Officer_Gender'] == 'M'

df_female_weapons = df[(filter_female) & (filter_weapons)]
df_male_weapons = df[(filter_male) & (filter_weapons)]

sns.set_palette('Reds_r')
fig, ax = plt.subplots(1,2)
fig.set_size_inches(17,8)

x_female_weapons = df_female_weapons.Weapon_Type.value_counts()[:5]
y_female_weapons = df_female_weapons.Weapon_Type.value_counts().index[:5]

x_male_weapons = df_male_weapons.Weapon_Type.value_counts()[:5]
y_male_weapons = df_male_weapons.Weapon_Type.value_counts().index[:5]

graph_female_weapons = ax[0].pie(x=x_female_weapons, labels=y_female_weapons, autopct='%1.2f%%')
graph_male_weapons = ax[1].pie(x=x_male_weapons, labels=y_male_weapons, autopct='%1.2f%%')

ax[0].set_title('Female Officer Weapon Found')
ax[1].set_title('Male Officer Weapon Found')

plt.show()
filter_male = df['Officer_Gender'] == 'M'
filter_stop_resolutions = df.Stop_Resolution != '-'

filter_female = df['Officer_Gender'] == 'F'

df_male_weapons = df[(filter_male) & (filter_stop_resolutions)]

df_female_weapons = df[(filter_female) & (filter_stop_resolutions)]

sns.set_palette('Reds_r')
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(17,8)

x_male_weapons = df_male_weapons.Stop_Resolution.value_counts()[:5]
y_male_weapons = df_male_weapons.Stop_Resolution.value_counts().index[:5]

x_female_weapons = df_female_weapons.Stop_Resolution.value_counts()[:5]
y_female_weapons = df_female_weapons.Stop_Resolution.value_counts().index[:5]

graph_female_weapons = ax[1].pie(x=x_female_weapons, labels=y_female_weapons, autopct='%1.2f%%')
graph_male_weapons = ax[0].pie(x=x_male_weapons, labels=y_male_weapons, autopct='%1.2f%%')

ax[0].set_title('Female Officer Stop Resolution')
ax[1].set_title('Male Officer Stop Resolution')

plt.show()
#chart config
sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_squads = df.Officer_Squad.value_counts()[:5]
labels_squads = df.Officer_Squad.value_counts().index[:5]

graph_squads = ax.pie(x=x_squads, labels=labels_squads, autopct='%1.2f%%')

plt.show()
sns.set_palette('Blues')
fig, ax = plt.subplots()
fig.set_size_inches(15,12)

x_squads = df.Officer_Squad.value_counts().index[:20]
y_squads = df.Officer_Squad.value_counts()[:20]

graph_squads = sns.barplot(x=x_squads, y=y_squads, data=df )

for item in graph_squads.get_xticklabels():
    item.set_rotation(90)

plt.show()
filter_squad_precinct = (df.Officer_Squad.isin(df.Officer_Squad.value_counts()[:20].index.tolist())) & (df.Precinct != 'Unknown')
df_squads_precinct = df[filter_squad_precinct]

#chart config
sns.set_palette('Greens')
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

x_squads_precinct = df_squads_precinct['Precinct'].value_counts()
labels_squads_precinct = df_squads_precinct['Precinct'].value_counts().index

graph_squads_precinct = ax.pie(x=x_squads_precinct, labels=labels_squads_precinct, autopct='%1.2f%%')

for item in graph_squads_precinct[0]:
    item.set_edgecolor('white')

plt.show()
df.head(2)
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

x_sector = df.Sector.value_counts().index
y_sector = df.Sector.value_counts()

graph_sectors = sns.barplot(x=x_sector, y=y_sector, data=df)

# for label in graph_sectors.get_xticklabels():
#     label.set_rotation(45)

plt.show()