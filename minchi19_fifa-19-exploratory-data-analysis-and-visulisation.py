import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_csv("../input/data.csv")
df.head()
df.info()
df.describe()
df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)
df.head()
def value_wage(df_value):

    try:

        value = float(df_value[1:-1])

        end = df_value[-1:]



        if end == 'M':

            value = value * 1000000

        elif end == 'K':

            value = value * 1000

    except ValueError:

        value = 0

        

    return value



df['Value'] = df['Value'].apply(value_wage)

df['Wage'] = df['Wage'].apply(value_wage)
print('Total number of countries : {0}'.format(df['Nationality'].nunique()))

print(df['Nationality'].value_counts().head(5))

print('Total number of clubs : {0}'.format(df['Club'].nunique()))

print(df['Club'].value_counts().head(5))
best_col=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

i=0

while i < len(best_col):

    print('Best {0} : {1}'.format(best_col[i],df.loc[df[best_col[i]].idxmax()][1]))

    i += 1
print('Most valued player : '+str(df.loc[df['Value'].idxmax()][1]))

print('Highest earner : '+str(df.loc[df['Wage'].idxmax()][1]))
df.sort_values(by = 'Value' , ascending = False)[['Name' , 'Club' , 'Nationality' , 

                                                     'Overall' , 'Value' , 'Wage']].head(5)
sns.countplot(y = 'Nationality', data=df, order = df.Nationality.value_counts().iloc[:10].index);
mean_wage_per_age = df.groupby('Age')['Wage'].mean()

pl = sns.barplot(x = mean_wage_per_age.index, y = mean_wage_per_age.values)

sns.lmplot(data = df, x = 'Age', y = 'SprintSpeed',lowess=True,scatter_kws={'alpha':0.01, 's':5,'color':'green'}, 

           line_kws={'color':'violet'})
talents_df = df.filter(["Name", "Club", "Overall"])

talents_df = talents_df[talents_df.Overall >= 85]

talents_df = talents_df.groupby("Club").count()

talents_df = talents_df.sort_values(by=['Name'],ascending=False)

top_ten_clubs = talents_df[:10]

top_ten_clubs = list(top_ten_clubs.index.values)



top_count =  list(talents_df.iloc[:, 1])

top_ten_clubs
#top_ten_clubs = ['FC Barcelona', 'Real Madrid', 'Manchester City', 'Arsenal', 'Liverpool', 'Manchester United', 'Borussia Dortmund', 'FC Bayern München', 'Juventus', 'Paris Saint-Germain']

top_ten_clubs_data = df.loc[df['Club'].isin(top_ten_clubs), :]

plt.figure(figsize=(8,6))

ax = sns.barplot(x = "Club",y = "Wage",data = top_ten_clubs_data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
sns.set(style="white")

plt.figure(figsize=(11,8))

p = sns.boxplot(x = 'Club', y = 'Value', data = top_ten_clubs_data)

p = plt.xticks(rotation=90)
top_ten_countries = df['Nationality'].value_counts().head(10).index.values

top_ten_countries_data = df.loc[df['Nationality'].isin(top_ten_countries), :]



plt.figure(figsize=(11, 8))

p = sns.boxplot(x = 'Nationality', y = 'Overall', data = top_ten_countries_data)
sns.jointplot(x=df['Age'],y=df['Potential'],

              joint_kws={'alpha':0.1,'s':5,'color':'m'},

              marginal_kws={'color':"m"})

 

df[df['Preferred Foot'] == 'Left'][['Name','Overall']].head()
 

df[df['Preferred Foot'] == 'Right'][['Name','Overall']].head()
value = df.Value

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)



sns.regplot(y="Age", x="Potential",  

        

             data=df);
plt.figure(figsize=(50,40))

p = sns.heatmap(df.corr(), annot=True, cmap = "summer")