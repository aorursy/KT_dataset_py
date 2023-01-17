# #Step 1. Import the necessary libraries



import pandas as pd

import numpy as np

#Step 2. Import the dataset from this address.

#Step 3. Assign it to a variable called euro12.¶



euro12 = pd.read_csv('../input/Euro 2012 stats TEAM.csv')



euro12
#Step 4. Select only the Goal column.



euro12.Goals



#OR



euro12['Goals']

#Step 5. How many team participated in the Euro2012?



euro12.shape[0] #16 teams participated
#Step 6. What is the number of columns in the dataset?



euro12.shape[1]



#OReuro12.info()
#Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline



discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]



discipline
#Step 8. Sort the teams by Red Cards, then to Yellow Cards



discipline.sort_values(['Yellow Cards', 'Red Cards'], ascending = False)
#Alternatively

c = euro12.groupby('Team').sum()

c = c.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)

c.iloc[:, -5:-3]
#OR

c = euro12.groupby('Team').sum()

c = c.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)

c.loc[:, ['Red Cards','Yellow Cards']]
#Step 9. Calculate the mean Yellow Cards given per Team



round(euro12['Yellow Cards'].mean())
#tep 10. Filter teams that scored more than 6 goals



euro12[euro12.Goals>6]
#Step 11. Select the teams that start with G

euro12[euro12.Team.str.startswith('G')]
#Step 12. Select the first 7 columns



euro12.iloc[:, :7]
#Step 13. Select all columns except the last 3



euro12.iloc[:, :-3]
#Step 14. Present only the Shooting Accuracy from England, Italy and Russia



euro12.loc[euro12.Team.isin(['Englad', 'Italy', 'Russia']), ['Shooting Accuracy']]