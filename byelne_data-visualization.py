

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from IPython.display import Image, display

print("Let us explore State Data")

# Load 'State_time_series.csv' 
df_state = pd.read_csv('../input/State_time_series.csv')

df_state.head(5)
#Change in per square feet value of houses across US
df_state.Date = pd.to_datetime(df_state.Date)
df_state.groupby(df_state.Date.dt.year)['ZHVIPerSqft_AllHomes'].mean().plot(kind='bar', figsize=(10, 6))
plt.suptitle('Median Per Square Feet value over the years', fontsize=12)
plt.ylabel('Dollars Per square Feet')
plt.xlabel('Year')
plt.show()

#Per square feet value for each month map(lambda x: x.strftime('%Y-%m-%d'))
df_state_time_sr.Date = pd.to_datetime(df_state_time_sr.Date)
month_year = df_state_time_sr.Date.map(lambda x: x.strftime('%m'))
df_state_time_sr.groupby(month_year)['ZHVIPerSqft_AllHomes'].mean().plot(kind='bar', figsize=(10,6))
plt.suptitle('Median Per Square Feet Value for each month', fontsize=12)
plt.ylabel('Dollars Per square Feet')
plt.xlabel('Month')
plt.show()
print("The per square feet value was high in 2007 and 2008. It started dropping after 2008, bottomed out in 2012, and started going up again from 2013.")
#Let us check how per square feet varies for states

#group by states. Calculate mean of per square foot and sort
psf_states = df_state.groupby([df_state.RegionName])['ZHVIPerSqft_AllHomes'].mean().sort_values(ascending=False)

psf_states.plot(kind='bar', figsize=(15, 6))
plt.suptitle('Median Per Square Feet value for All states', fontsize=12)
plt.ylabel('Dollars Per square Feet')
plt.xlabel('State')
plt.show()


fig,ax = plt.subplots()

states = ["DistrictofColumbia","Texas","WestVirginia","California"]
#group by states and year. Calculate mean of per square foot and sort
for state in states:
  
    state_list = (df_state.loc[df_state['RegionName'] == state])
     
    group_by_year = state_list.groupby([state_list.Date.dt.year])['ZHVIPerSqft_AllHomes'].mean()
   
    
    group_by_year.plot(kind='line', figsize=(15, 6), label=state)
     #ax.plot(group_by_year, label=state)
    

ax.set_xlabel("year")
ax.set_ylabel("Per Sq FT")
ax.legend(loc='best')

plt.show