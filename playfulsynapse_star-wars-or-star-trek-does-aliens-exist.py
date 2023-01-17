import pandas as pd

#import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from matplotlib_venn import venn3 



#ignore some anoying warnings from the seaborn plot

import warnings

warnings.filterwarnings('ignore')
#Read the data

whole_dataset=pd.read_csv('../input/2016 Stack Overflow Survey Responses.csv')
whole_dataset.head()
#Lets look at the columns available ...

print(whole_dataset.columns)
#Extract the Star Wars Fans, and convert the index to a set

StarWars=whole_dataset[whole_dataset['star_wars_vs_star_trek']=='Star Wars']

StarWars_set = set(StarWars.index)
#Like wise for the Star Trek

StarTreck=whole_dataset[whole_dataset['star_wars_vs_star_trek']=='Star Trek']

StarTreck_set = set(StarTreck.index)

Aliens=whole_dataset[whole_dataset['aliens']=='Yes'] #assumes NaA = NO

Aliens_set =set(Aliens.index)
venn3([StarWars_set, StarTreck_set, Aliens_set], ('Star Wars', 'Star Treck', 'Aliens exists'))

plt.show()
#Lets calculate the percentages...

StarWars_believers =StarWars_set.intersection(Aliens_set)

print('Total number of Star Wars fans        : ',len(StarWars_set) )

print('percentage who believes Aliens exists :', round(len (StarWars_believers)/len(StarWars_set),3))
StarTreck_believers =StarTreck_set.intersection(Aliens_set)

print('Total number of Star Treck fans       : ',len(StarTreck_set) )

print('percentage who believes Aliens exists :', round(len (StarTreck_believers)/len(StarTreck_set),3))