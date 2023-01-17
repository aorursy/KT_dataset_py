import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pokemonData = pd.read_csv('../input/Pokemon.csv')

pokemonData.head()
pokemonData.columns
Column_to_group_by='Type 1'

Columns_to_include=['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed','Legendary']

groups=pokemonData.groupby(Column_to_group_by)[Columns_to_include]

meanValues=groups.mean()

levelCount=groups.count()

print(meanValues)

print('--------------------------------------------------------------------------------------------')

print(levelCount)
merged_frame = pd.concat([levelCount['Total'], meanValues], axis=1)



# add new column names to the merged dataframe

new_col_names=['NumRows','TotalPoints', 'HP', 'Attack', 'Defense','SpecialAttack', 'SpecialDefence', 'Speed','%Legendary']

merged_frame.columns=new_col_names



# sort the results by the Total_points

merged_frame.sort_values("TotalPoints",ascending=False,inplace=True)

merged_frame
#Start with rounding to 2 decimals

merged_frame.style.format("{:.2f}")


#Add that the %Legend colums should be printed as percentages with one decimal

merged_frame.style.format("{:.2f}").format({'%Legendary': '{:.1%}'})
#And now add some colour coding to make the table easier to read.

cmap=sns.diverging_palette(250, 5, as_cmap=True)



merged_frame.style.format("{:.2f}").format({'%Legendary': '{:.1%}'}).background_gradient(cmap, axis=0)