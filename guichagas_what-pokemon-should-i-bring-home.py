import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl 

import matplotlib.pyplot as plt

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#import to dataframe

df = pd.read_csv("../input/pokemon.csv")
#let's see what we have here

df.columns
# a quick standard check 

df.describe()
df.head()
#Let's great a df to plot our answer



df1 = df.loc[:,('hp','generation')]

df2 = df.loc[:,('attack','generation')]

df3 = df.loc[:,('defense','generation')]

df4 = df.loc[:,('speed','generation')]



df1['attribute'] = 'hp'

df1.rename(columns={'hp': 'value'}, inplace=True)



df2['attribute'] = 'attack'

df2.rename(columns={'attack': 'value'}, inplace=True)



df3['attribute'] = 'defense'

df3.rename(columns={'defense': 'value'}, inplace=True)



df4['attribute'] = 'speed'

df4.rename(columns={'speed': 'value'}, inplace=True)



frames = [df1, df2, df3, df4]



result = pd.concat(frames)
# plotting

sns.factorplot(kind='box',        # Boxplot

               y='value',         # Y-axis - values for boxplot

               x='attribute',     # X-axis - first factor

               hue='generation',  # Second factor denoted by color

               data=result,       # Dataframe 

               size=10,           # Figure size (x100px)      

               aspect=1.6,        # Width = size * aspect 

               legend_out=False)  # Make legend inside the plot

plt.title('Attribute variation between Pokemon generations')
#Preparing df for plotting the answer



df5 = df.loc[:,('type1','generation', 'name')]

df5 = df5.groupby(['type1', 'generation'], as_index = False).count()

#df5 = df5.pivot(index='generation', columns='type1', values='name')



df5 = pd.pivot_table(df5, 

                       index='generation', 

                       columns='type1', 

                       values='name', 

                       aggfunc=np.sum,

                       fill_value = 0,

                       margins=True)



# this is a dataframe counting pokemons according to their type1 and generation
# now we will create a % of the value in df5



df6 = df5.div(df5['All'], axis='index') 

df6.drop('All', axis=1, inplace = True)

df6.drop('All', inplace = True)

df6 = df6*100



df6 = df6.T
# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(df6, annot=True, linewidths=.5, ax=ax)



plt.title('Distribution of Pokemon type within each generation')
# Pre preparation of data



ab_df = df[['type1', 'abilities']].copy()



t = ab_df['abilities'].astype(str).str.replace('[', '')

t = t.str.replace(']', '')

t = t.str.replace("'", '')



temp_abilities = t.str.split(',', expand=True)



temp_abilities['type'] = ab_df['type1']



temp_abilities



#stack columns

df_abilities_0 = temp_abilities[['type', 0]].copy()

df_abilities_1 = temp_abilities[['type', 1]].copy()

df_abilities_2 = temp_abilities[['type', 2]].copy()

df_abilities_3 = temp_abilities[['type', 3]].copy()

df_abilities_4 = temp_abilities[['type', 4]].copy()

df_abilities_5 = temp_abilities[['type', 5]].copy()



df_abilities_0[['type', 'ability']] = temp_abilities[['type', 0]]

df_abilities_1[['type', 'ability']] = temp_abilities[['type', 1]]

df_abilities_2[['type', 'ability']] = temp_abilities[['type', 2]]

df_abilities_3[['type', 'ability']] = temp_abilities[['type', 3]]

df_abilities_4[['type', 'ability']] = temp_abilities[['type', 4]]

df_abilities_5[['type', 'ability']] = temp_abilities[['type', 5]]



df_abilities = pd.concat([df_abilities_0, df_abilities_1, df_abilities_2,

                         df_abilities_3, df_abilities_4, df_abilities_5],

                        ignore_index=True)



df_abilities = df_abilities[['type', 'ability']]



df_abilities = df_abilities[df_abilities['ability'].notnull()]





# Overall abilities



over_ability = df_abilities.groupby(['ability'], 

                                    as_index=False).count().sort_values(by=['type'],

                                                                        ascending=False).reset_index()



print('From ' + str(over_ability['ability'].size) + ' abilities, the top 5 more common are (in order):'

     + str(over_ability['ability'][0:5].values))