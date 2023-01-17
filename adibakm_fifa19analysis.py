import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/data.csv')

df.head()



print(df.shape) #18207 rows and 89 columns
def clean_string(s):

    try:

        s = s.replace('€', '')

        s=s.replace('M','')

        s=s.replace('K','')

    except:

        pass



    return s

df['International Reputation'] = df['International Reputation'].fillna(df['International Reputation'].mean()).astype(int)

df['Crossing'] = df['Crossing'].fillna(df['Crossing'].mean()).astype(int)

df['Finishing'] = df['Finishing'].fillna(df['Finishing'].mean()).astype(int)

df['HeadingAccuracy'] = df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean()).astype(int)

df['ShortPassing'] = df['ShortPassing'].fillna(df['ShortPassing'].mean()).astype(int)

df['Release Clause'] = df['Release Clause'].apply(clean_string).astype(float)

df['Release Clause'] = df['Release Clause'].fillna(df['Release Clause'].mean()).astype(int)

df['Wage'] = df['Wage'].apply(clean_string).astype(float)

df['Wage'] = df['Wage'].fillna(df['Wage'].mean()).astype(int)

df['Value'] = df['Value'].apply(clean_string).astype(float)

df['Value'] = df['Value'].fillna(df['Value'].mean()).astype(int)

df['Weak Foot'] = df['Weak Foot'].fillna(df['Weak Foot'].mean()).astype(int)

df['Skill Moves'] = df['Skill Moves'].fillna(df['Skill Moves'].mean()).astype(int)

df['Volleys'] = df['Volleys'].fillna(df['Volleys'].mean()).astype(int)

df['Dribbling'] = df['Dribbling'].fillna(df['Dribbling'].mean()).astype(int)

df['Curve'] = df['Curve'].fillna(df['Curve'].mean()).astype(int)

df['FKAccuracy'] = df['FKAccuracy'].fillna(df['FKAccuracy'].mean()).astype(int)

df['LongPassing'] = df['LongPassing'].fillna(df['LongPassing'].mean()).astype(int)

df['BallControl'] = df['BallControl'].fillna(df['BallControl'].mean()).astype(int)

df['Acceleration'] = df['Acceleration'].fillna(df['Acceleration'].mean()).astype(int)

df['SprintSpeed'] = df['SprintSpeed'].fillna(df['SprintSpeed'].mean()).astype(int)

df['Agility'] = df['Agility'].fillna(df['Agility'].mean()).astype(int)

df['Reactions'] = df['Reactions'].fillna(df['Reactions'].mean()).astype(int)

df['Balance'] = df['Balance'].fillna(df['Balance'].mean()).astype(int)

df['ShotPower'] = df['ShotPower'].fillna(df['ShotPower'].mean()).astype(int)

df['Jumping'] = df['Jumping'].fillna(df['Jumping'].mean()).astype(int)

df['Stamina'] = df['Stamina'].fillna(df['Stamina'].mean()).astype(int)

df['Strength'] = df['Strength'].fillna(df['Strength'].mean()).astype(int)

df['LongShots'] = df['LongShots'].fillna(df['LongShots'].mean()).astype(int)

df['Aggression'] = df['Aggression'].fillna(df['Aggression'].mean()).astype(int)

df['Interceptions'] = df['Interceptions'].fillna(df['Interceptions'].mean()).astype(int)

df['Positioning'] = df['Positioning'].fillna(df['Positioning'].mean()).astype(int)

df['Vision'] = df['Vision'].fillna(df['Vision'].mean()).astype(int)

df['Penalties'] = df['Penalties'].fillna(df['Penalties'].mean()).astype(int)

df['Composure'] = df['Composure'].fillna(df['Composure'].mean()).astype(int)

df['Marking'] = df['Marking'].fillna(df['Marking'].mean()).astype(int)

df['StandingTackle'] = df['StandingTackle'].fillna(df['StandingTackle'].mean()).astype(int)

df['SlidingTackle'] = df['SlidingTackle'].fillna(df['SlidingTackle'].mean()).astype(int)        

df['GKDiving'] = df['GKDiving'].fillna(df['GKDiving'].mean()).astype(int)

df['GKHandling'] = df['GKHandling'].fillna(df['GKHandling'].mean()).astype(int)

df['GKKicking'] = df['GKKicking'].fillna(df['GKKicking'].mean()).astype(int)

df['GKPositioning'] = df['GKPositioning'].fillna(df['GKPositioning'].mean()).astype(int)

df['GKReflexes'] = df['GKReflexes'].fillna(df['GKReflexes'].mean()).astype(int)
print(df.info())
print(df.columns)
df.describe()
df.describe(include=['object'])
df['Nationality'].value_counts()
df['Nationality'].value_counts(normalize=True)
df.sort_values(by='International Reputation', ascending=False).head() # dataset arranged according to international reputation

df.sort_values(by='Height', ascending=False).head() #multiple columns can also be sorted by='blah blah blah' asending = 3 boolean values
df['Overall'].mean()
df[df['Overall'] == 90].mean()
df[df['Overall'] == 90]['Penalties'].mean() #guys with 90 overall aren't that good on fifa at penalties
df[(df['Age'] == 25) & (df['Nationality'] == 'Brazil')]['Overall'].max()
df.loc[0:10, 'Skill Moves':'Position'] #this is strange, we get Messi, Ronaldo for body type clearly we have some work to do
df.iloc[0:5, 7:10]
df[-1:]
 #df.apply(np.max) 
df[df['Nationality'].apply(lambda state: state[0] == 'B')].head()
#d = {'No' : False, 'Yes' : True}

#df['International plan'] = df['International plan'].map(d)

#df.head()

#df = df.replace({'Voice mail plan': d})

#df.head() replacing columns
columns_to_show = ['Overall', 'Age', 

                   'Nationality']



df.groupby(['Position'])[columns_to_show].describe(percentiles=[])
columns_to_show = ['Overall', 'Age', 

                   ]



df.groupby(['Position'])[columns_to_show].agg([np.mean, np.std, np.min, 

                                            np.max])
pd.crosstab(df['Overall'], df['Age']) #,normalize=True for percentage
#df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],

#              ['Area code'], aggfunc='mean')

#This will resemble pivot tables to those familiar with Excel. And, of course, pivot tables are implemented in Pandas:

#the pivot_table method takes the following parameters:

#values – a list of variables to calculate statistics for,

#index – a list of variables to group data by,

#aggfunc – what statistics we need to calculate for groups, ex. sum, mean, maximum, minimum or something else.
physical_rating = df['Acceleration']+ df['SprintSpeed']+ df['Agility']+ df['Reactions']+df['Balance'] /5

df.insert(loc=len(df.columns), column='Physical rating', value=physical_rating) 

# loc parameter is the number of columns after which to insert the Series object

# we set it to len(df.columns) to paste it at the very end of the dataframe

df.head()
# Matplotlib forms basis for visualization in Python

import matplotlib.pyplot as plt



# We will use the Seaborn library

import seaborn as sns

sns.set()



# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg'
features = ['Crossing', 'Finishing']

df[features].hist(figsize=(10, 4));
features=['Composure','Penalties']

df[features].plot(kind='density', subplots=True, layout=(1, 2), 

                  sharex=False, figsize=(10, 4));
sns.distplot(df['Age']); #this needs to be an int
sns.boxplot(x='Special', data=df);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))

sns.boxplot(data=df['Volleys'], ax=axes[0]);

sns.violinplot(data=df['Volleys'], ax=axes[1]);
numerical = list(set(df.columns) - 

                 set(['GKDiving', 'Interceptions', 'Volley', 

                      'Age', 'StandingTackle', 'Crossing', 'Release Claue', 'ID', 'Unnamed: 0','SlidingTackle','Marking', 'Positioning','GKPositioning','LongPassing','Jumping','Potential','Overall','Curve','ShortPassing','Jersey Number', 'HeadingAccuracy','LongShots','Special','Wage','Skill Moves','Value','Volleys','ShotPower','GKHandling','GKReflexes','FKAccuracy','GKKicking','Release Clause','Heading Accuracy']))



# Calculate and plot

corr_matrix = df[numerical].corr()

sns.heatmap(corr_matrix);
plt.scatter(df['International Reputation'], df['Value']);
#sns.jointplot(x='International Reputation', y='Value', 

#              data=df, kind='scatter');
sns.jointplot('Wage', 'Release Clause', data=df,

              kind="kde", color="g");
# `pairplot()` may become very slow with the SVG format

%config InlineBackend.figure_format = 'png'

sns.pairplot(df[numerical]);
%config InlineBackend.figure_format = 'svg'
sns.lmplot('Value', 'International Reputation', data=df, hue='Nationality', fit_reg=False);