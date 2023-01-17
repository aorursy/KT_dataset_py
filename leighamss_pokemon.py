import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.rcParams['figure.figsize'] = 12, 6

poke = pd.read_csv('../input/Pokemon.csv')
poke.head()
poke.index = poke['Name']
poke.isnull().any()
percent_of_NAN_values = 0

amount = len(poke)

for i in poke['Type 2']:

    if pd.isnull(i):

        percent_of_NAN_values+=1

percent_of_NAN_values =(percent_of_NAN_values/amount) * 100



'The percentage of values in the type 2 column that are null is % {}'.format(percent_of_NAN_values)
"Seems like there is a lot of missing data in that portion of the dataset"

poke.drop('Type 2',axis =1,inplace =True )
#Using a lambda expression and summing the different attributes for each pokemon to obtain the value

poke['Median'] = poke['Name'].apply(lambda x:np.median(poke.loc[x][['Attack','Defense','Sp. Atk','Sp. Def','Speed','HP']]))



'The Score for the most well rounded pokemon is {0} which belongs to {1} '.format(poke['Median'].max(),poke[poke['Median']==130]['Name'][0]+'is')

sns.countplot(data=poke,x ='Type 1',orient='v')

plt.title('Count of different pokemon types',size = 20)

plt.show()


plt.title('Generation x Type',size = 20)

sns.heatmap(poke.pivot_table(values='Total',index='Generation',columns='Type 1'),cmap='coolwarm')

plt.show()
"""As depicted in the heat map above in some generations certain types of pokemon 

are much more common that in others"""

newgen = 0

for gen in range(0,len(poke['Generation'].value_counts())+1):

    if gen > 0:

        print('Most common type in Generation {0}: {1} \n'.format(newgen, poke[poke['Generation']==gen]['Type 1'].value_counts().head(1)))

    newgen+=1

    
gen_by_type = [{'Top 3 Most Common Types(descending)': 'Water', 'Number Of Times Most Popular': 3, },

         {'Top 3 Most Common Types(descending)': 'Normal',  'Number Of Times Most Popular': 2,},

         {'Top 3 Most Common Types(descending)': 'Ghost',  'Number Of Times Most Popular': 1,}]

df = pd.DataFrame(gen_by_type)

df
plt.title('Total points x Type',size = 20)

sns.swarmplot(data=poke,x = 'Type 1',y = 'Total',hue = 'Legendary',palette='BrBG')

plt.show()