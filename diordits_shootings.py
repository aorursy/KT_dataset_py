import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')

df
df['Age_class'] = df['age'].apply(lambda x: (x//10)*10)

df2 = df[['race', 'Age_class']].groupby(by=['race','Age_class']).size().unstack().fillna(0)
df2.loc[['Black', 'White', 'Hispanic']].apply(lambda x: (x/sum(x))*100, axis=1).unstack().unstack().plot(title='distribution within the national')

df2.loc[['Black', 'White', 'Hispanic']].unstack().unstack().plot(title='distribution over count accidents')

df2.loc[['Black', 'White', 'Hispanic']].apply(lambda x: (x/sum(x))*100).unstack().unstack().plot(title='distribution within age class  ')
mantel_accidents = df[df['signs_of_mental_illness']==1].groupby(['date','signs_of_mental_illness']).size().unstack().reset_index()

mantel_accidents['date'] = pd.to_datetime(mantel_accidents['date'] )

mantel_accidents
mantel_accidents['seazon'] = mantel_accidents['date'].apply(lambda x: x.month)

mantel_accidents.groupby(['seazon', True]).size().unstack().fillna(0).apply(lambda x: sum(x), axis=1).plot()


mantel_accidents.groupby(['seazon', True]).size().unstack().fillna(0).plot()