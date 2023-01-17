import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

draft_years = []
for year in range(2009, 2019):
    df = pd.read_csv("../input/{}.csv".format(year))    
    draft_years.append(df)
    
last_10_years = pd.concat(draft_years)

# Fill NA values that we'll do math on later
last_10_years = last_10_years.fillna({'GP': 0.0})

# Cleanup for moved/renamed teams.
last_10_years = last_10_years.replace('Phoenix Coyotes', 'Arizona Coyotes')
last_10_years = last_10_years.replace('Atlanta Thrashers', 'Winnipeg Jets')
teams = last_10_years.groupby('Team')['GP'].agg('sum').sort_values()

plt.figure(figsize=(15,8))
teams.plot(kind='barh', title='Team Draft Picks by Total NHL Games Played')
plt.show()