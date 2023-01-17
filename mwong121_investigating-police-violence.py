import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import missingno as msno
# ../input/data-police-shootings/fatal-police-shootings-data.csv
df = pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv")
df.info()
df.describe(include=['O'])
df.nunique()
df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
df['year'] = pd.to_datetime(df['date']).dt.strftime('%Y')
df['month'] = pd.to_datetime(df['date']).dt.strftime('%m')
d = {'W' :'White','B' :'Black','H' :'Hispanic'
     ,'A' :'Asian','N' :'Native American','O' :'Other'}
df['race'] =  df['race'].map(d)

### What is the average age of the victim by race?
sns.distplot(df['age'],kde=False)
df[['race', 'age']].groupby(['race'], as_index=False).mean()
plt.figure(figsize=(8,5))
sns.violinplot(x='race',y='age',data=df,hue='gender',split=True)
### Noticable bump in females shot in both Native American and Asian females.
plt.figure(figsize=(8,4))
sns.stripplot(x='race',y='age',data=df,hue='manner_of_death')
### most of the record incidents involved shootings
plt.figure(figsize=(8,5))
sns.stripplot(x='race',y='age',data=df)
### The youngest and oldest victims of police shooting were white.
plt.figure(figsize=(16,5))
sns.countplot(x=df['month'],order = df['month'].value_counts().index)
### Top months where shooting were seen are in the beginning of the year. What could be causing this?
### Whites make up about 50% of victims of police brutality
plt.pie(df['race'].value_counts(),autopct='%1.1f%%')
plt.title('Death by Race')
plt.axis('equal')
plt.show()

df.isnull().sum()
msno.matrix(df)
missing_percentage=df.isna().sum()*100/df.shape[0]
missing_percentage
df.dropna(inplace=True)
cardinality ={}
for col in df.columns:
    cardinality[col] = df[col].nunique()
cardinality
#Since there are 90 different types of weapons listed, 
#some weapons listed are not life threatening such as 'barstool' 
#and so I am re-classifying weapons between life threatening weapons and non-harmful objects.
df['armed'].unique()
l = ['gun', 'pole and knife', 'box cutter', 'flagpole', 'metal pole', 'baseball bat and fireplace poker', 
     'gun and knife', 'chain saw', 'hatchet and gun', 'crowbar', 'fireworks', 'pellet gun', 'samurai sword', 
     'vehicle and gun', 'grenade', 'Airsoft pistol', 'ice pick', 'machete and gun', 'baseball bat and bottle', 
     'BB gun', 'gun and vehicle', 'vehicle and machete', 'knife and mace', 'hatchet', 'guns and explosives', 
     'Taser', 'gun and sword', 'incendiary device', 'sword', 'crossbow', 'spear', 'bayonet', 'machete', 'knife', 
     'pick-axe', 'gun and car', 'vehicle', 'motorcycle', 'car', 'bow and arrow','unknown weapon', 'BB gun and vehicle', 
     'baseball bat and knife', 'claimed to be armed']
df['true_weapon'] = np.where(df['armed'].apply(lambda x: any([k in str(x) for k in l])),'weapons','non_weapons')
df.head()
### if we remove all factors that can make a victim look threatening, maybe we can identify those deaths that were lead by racial bias.
df['threatening_appearance'] = np.where( ( (df['signs_of_mental_illness'] == True) | (df['threat_level'] == 'attack' ) | (df['flee'] == 'Car' ) |  (df['true_weapon'] == 'weapons' ) ) , 'Appears Threatening', 'No Threat')
df.tail()
# df.shape
innocent = df[df["threatening_appearance"]=='No Threat']
innocent.tail()
# innocent.shape
### The gap in percentage of deaths is closing in when looking at the 'innocent' sample
labels = ['White', 'Black', 'Hispanic', 'Asian','Native','Other']
plt.pie(innocent['race'].value_counts(),autopct='%1.1f%%',labels=labels)
plt.title('Innocent - Death by Race')
plt.axis('equal')
plt.show()
prop = pd.crosstab(df['race'],df['threatening_appearance'])
# prop = prop.sort_values('No Threat', ascending=False)
prop = prop.nlargest(6, 'No Threat')
prop
# plt.figure(figsize=(10,5))
# ax = sns.countplot(x='race', hue="threatening_appearance", data=df)
### Drawing a distinction in percentage of deaths on 'non-threating' victims. Innocent Blacks and Hispanics see the largest proportion of violence from police.
ax = (prop.div(prop.sum(1), axis=0)).plot(kind='bar',figsize=(15,6),width = 0.8)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01))
plt.legend(title='Has Threatening Appearance',loc=9, bbox_to_anchor=(.5, -0.35))

plt.title("Percentage of Deaths of Non Theatening Persons", y=1.013, fontsize=22)


plt.xticks(rotation=45)
plt.xlabel(xlabel = 'Race', labelpad=16)
plt.ylabel(ylabel = 'Count [Deaths]', labelpad=16)
plt.show()
### No strong relationships were found
df['Has_threatening_appearance'] = np.where(df['threatening_appearance'] != 'No Threat',True,False)
df['weapon'] = np.where(df['true_weapon'] == 'weapons',True,False)
df['fled'] = np.where(df['flee'] != 'Not fleeing',True,False)
df['threat'] = np.where(df['threat_level'] == 'attack',True,False)
selected_columns = df[["weapon","fled","body_camera","signs_of_mental_illness","race","Has_threatening_appearance"]]
new_df = selected_columns.copy()
sns.heatmap(new_df.corr(),cmap='Blues')
new_df
### Deaths of innocent White Americans seem to be decrease over the years, while deaths of Black Americans seem to hover in the 200's over the past 5 years
chart = sns.catplot(
    data=df[df['race'].isin(['Black', 'White'])],
    x='year',
    kind='count',
    palette='Set1',
    col='race',
    aspect=1,
)
chart.set(xlabel='Year of Incident', ylabel='Count of Deaths', title='Innocent Deaths by Race')
chart.set_xticklabels(rotation=65, horizontalalignment='right')

### Top states with deaths
state_count = df.groupby("state")
state_count = state_count["id"].count()
state_count = state_count.reset_index().rename(columns={"id":"count"}).sort_values(by = 'count', ascending=False)
state_count.head(10)
### Top 5 states experiencing police violence
state_year_count = df.groupby(["year","state"])
state_year_count = state_year_count["id"].count()
state_year_count = state_year_count.reset_index().rename(columns={"id":"count"}).sort_values(by = 'count', ascending=False)
top_3_state_year_count = state_year_count.loc[state_year_count.groupby('year')['count'].nlargest(3).reset_index()['level_1']]
top_3_state_year_count

### The state of California has the highest count of incidents when breaking down by year.
sns.catplot(x = "year",      
            y = "count",      
            hue = "state", 
            data = top_3_state_year_count,     
            kind = "bar")
!pip install chart_studio

import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

dfw = df.state.value_counts().reset_index()

df2 = dict(type='choropleth',
            locations = dfw['index'],
            locationmode = 'USA-states',
            colorscale = 'reds',
            z = dfw['state'],
            colorbar = {'title':"Death Count"}
            )
layout = dict(title = 'Police Violence in the US',
              geo = dict(scope='usa')
             )
choromap = go.Figure(data = [df2],layout = layout)
iplot(choromap)
### What is the percentage of incident against general population?
state_year_count = df.groupby(["year","state","race"])
state_year_count = state_year_count["id"].count()
state_year_count = state_year_count.reset_index().rename(columns={"id":"count"}).sort_values(by = 'count', ascending=False)
top_5_state_year_race = state_year_count.loc[state_year_count.groupby('year')['count'].nlargest(5).reset_index()['level_1']]
top_5_state_year_race.head()

df_population = pd.DataFrame({'race':['White','Black','Asian','Hispanic','Native American','Other'],'population':[0.601,0.134,0.059,0.185,0.013,0.008]})

df_population
state_pop = pd.read_csv("/kaggle/input/population-usa-2018/Population_Distribution_by_Race_2018.csv")
state_pop= state_pop.fillna(0)
state_pop["Other"] = state_pop["Native Hawaiian/Other Pacific Islander"]+state_pop["Two Or More Races"]
state_pop = state_pop.drop(['Total','Native Hawaiian/Other Pacific Islander','Two Or More Races'],axis=1)
state_pop = state_pop.rename(columns={"American Indian/Alaska Native":"Native American"})
state_pop.head()

state_2 = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

state_pop["Location"].replace(state_2, inplace=True)
state_pop.head()
whole_pop = state_pop.melt(id_vars=["Location"], 
        var_name="race", 
        value_name="population")

whole_pop.head()
new_df = top_5_state_year_race.merge(whole_pop, left_on=['state','race'], right_on=['Location','race']).drop('Location', 1)
new_df.head()
new_df['pop_targeted']= (new_df['count']/new_df['population'])*100
new_df.head()
### Do body cameras reduce police violence?
body_camera = pd.crosstab(df['body_camera'],df['threatening_appearance'])
body_camera.head()
### 6% of all violent acts from the police are incidents recorded by a body camera. We do not see a reduction in violence in police officers who wear body cameras.

ax = (body_camera.div(body_camera.sum(1), axis=0)).plot(kind='bar',figsize=(15,6),width = 0.8)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01))
plt.legend(title='Body Camera',loc=9, bbox_to_anchor=(.5, -0.35))

plt.xticks(rotation=45)
plt.xlabel(xlabel = 'Threat Appearance')
plt.show()