import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
titanic_df = pd.read_excel('../input/titanic3.xls', index_col=None, na_values=['NA'])
#to better understand the true extent of economic inequality, I have multiplied the fare by 23.45 to roughly adjust for inflation

titanic_df['fare_inf'] = titanic_df['fare']*23.45
titanic_df.head()
#Create a variable we can use to sort the ages in groups rather than as individual numbers.

group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
#A visualization of the survial rates based on age

titanic_df.groupby(group_by_age).mean().plot(y=['survived'], kind='bar', title='Survival Rate by Age Group')
titanic_df.groupby(['pclass']).mean()
titanic_df.groupby(['pclass']).mean().plot(y=['survived'], kind='bar', title='Survival Rate by Class')
#Create variable so we only look at the children in the dataframe who are younger than 10

children_df = titanic_df[(titanic_df['age']<=10)]
#it is worth noting that samples are very small. Especially as we divide the these data further

children_df.count()
#Similar to what we did for the passengers as a whole, we are simplifying the integers into groups

children_age_groups = pd.cut(children_df['age'],np.arange(0,11,2))

children_df.groupby(children_age_groups).mean()
#survival rates for children based on age range

children_df.groupby(children_age_groups).mean().plot(y=['survived'], kind='bar')
children_df.groupby(['pclass']).mean().plot(y='sibsp', kind='bar')
#survival rates of children based on class

children_df.groupby(['pclass']).mean().plot(y=['survived'], kind='bar')
#the sex of the child has very little effect on the survival rate of the child

children_df.groupby(['sex']).mean().plot(y=['survived'], kind='bar')
#there is a 33% difference in survival rate between children in 1st and 3rd class

children_df.groupby(['pclass']).mean()['survived'].iloc[0] - children_df.groupby(['pclass']).mean()['survived'].iloc[2]
titanic_df.groupby(['pclass']).mean()['survived'].iloc[0] - titanic_df.groupby(['pclass']).mean()['survived'].iloc[2]
titanic_rates = pd.cut(titanic_df["fare"], np.arange(0,600,50))
titanic_df[(titanic_df['fare'] == titanic_df['fare'].max())]
titanic_df.groupby(titanic_rates).mean().fillna(0).plot(y=['survived'], kind='line')