#Creating dictionary and list based on above data to be used in below questions.
import numpy as np

data = {'birds': ['Cranes', 'Cranes', 'plovers', 'spoonbills', 'spoonbills', 'Cranes', 'plovers', 'Cranes', 'spoonbills', 'spoonbills'], 
        'age': [3.5, 4, 1.5, np.nan, 6, 3, 5.5, np.nan, 8, 4], 
        'visits': [2, 4, 3, 4, 3, 4, 2, 2, 3, 2], 
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

import pandas as pd
birds_df = pd.DataFrame(data=data,index=labels)
birds_df
print(birds_df.info())
birds_df.head(2)
birds_df[['birds','age']]
birds_df.iloc[[2,3,7]][['birds', 'age', 'visits']]
birds_df[birds_df['visits'] < 4]
birds_df[birds_df['age'].isna()][['birds','visits']]
birds_df[(birds_df['birds'] == 'Cranes') & (birds_df['age'] < 4)]
birds_df[(birds_df['age'] >= 2) & (birds_df['age'] <= 4)]
birds_df[birds_df['birds'] == 'Cranes']['visits'].sum()
birds_df.groupby(['birds']).mean()['age']
birds_df.loc['k'] = ['NewBird',2.75,3,'yes']
print('Added a new row with index k \n', birds_df)

birds_df.drop('k',inplace=True)
print('\n Deleted the new row with index k \n',birds_df)

birds_df.groupby(birds_df['birds']).count()
birds_df.sort_values(by=['age','visits'],ascending=[False,True])
birds_df['priority'].replace(['yes','no'],[1,0],inplace=True)
birds_df
birds_df['birds'].replace('Cranes','trumpeters',inplace=True)
birds_df