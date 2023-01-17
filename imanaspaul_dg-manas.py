import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
# configure matplotlib to output inline
%matplotlib inline
# Rape victims
rape_victims = pd.read_csv('../input/20-victims-of-rape/20_Victims_of_rape.csv')

rape_victims.head()
# remove the total victims column
rape_victims = rape_victims[rape_victims['Subgroup'] != 'Total Rape Victims']

# let's check if the all the rape cases are reported
rape_victims[rape_victims['Victims_of_Rape_Total'] != rape_victims['Rape_Cases_Reported']].head()
rape_victims["Unreported_cases"] = rape_victims["Victims_of_Rape_Total"] - rape_victims["Rape_Cases_Reported"]

rape_victims[rape_victims["Unreported_cases"] > 0].head()
# Ploting the unreported cases by statewise
unreported_victims_by_state = rape_victims.groupby('Area_Name').sum()
# deleting no existing year row
unreported_victims_by_state.drop('Year', axis = 1, inplace = True)

plt.subplots(figsize = (15, 6))
ct = unreported_victims_by_state[unreported_victims_by_state['Unreported_cases'] 
                                 > 0]['Unreported_cases'].sort_values(ascending = False)

ax = ct.plot.bar()
ax.set_xlabel('Area Name')
ax.set_ylabel('Total Number of Unreported Rape Victims from 2001 to 2010')
ax.set_title('Statewise total Unreported Rape Victims throughout 2001 to 2010')
plt.show()
