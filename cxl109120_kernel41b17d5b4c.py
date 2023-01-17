import pandas as pd
Seasons_Stats = pd.read_csv("../input/Seasons_Stats cleaned.csv")
Seasons_Stats.columns
Seasons_Stats = Seasons_Stats [['Year', 'Player', 'Age',  'G',  'MP']]



# G = Game played per season

# MP = Minutes played per season
Seasons_Stats.head()
import numpy as np

Seasons_sum = Seasons_Stats.groupby('Year', as_index=False).agg({'G':'sum', 'MP':'sum'})

Seasons_sum.head()



Seasons_age = Seasons_Stats.groupby('Year', as_index=False).agg({'Age': 'min'})

Seasons_age.head()



Seasons_age.plot(kind='line',x='Year',y='Age', color='red')
Seasons_yearly = pd.DataFrame({'Year': Seasons_sum['Year'], 'Avg': Seasons_sum['MP']/Seasons_sum['G']})
Seasons_yearly.head()
Seasons_yearly.plot(kind='line',x='Year',y='Avg', color='red')
Seasons_std = Seasons_Stats.groupby('Year', as_index=False).agg({'G':'std', 'MP':'std'})

Seasons_std.head()
Seasons_std_yearly = pd.DataFrame({'Year': Seasons_std['Year'], 'Std': Seasons_std['MP']/Seasons_std['G']})

Seasons_std_yearly.head()
Seasons_std_yearly.plot(kind='line',x='Year',y='Std', color='green')