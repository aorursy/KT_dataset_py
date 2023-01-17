import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

draft_years = []
for year in range(2009, 2019):
    df = pd.read_csv("../input/{}.csv".format(year))
    
    df['LinearDraftValue'] = len(df) - df['Overall']
    
    draft_years.append(df)
    
last_10_years = pd.concat(draft_years)

# Fill NA values that we'll do math on later
last_10_years = last_10_years.fillna({'GP': 0.0})

# Cleanup for moved/renamed teams.
last_10_years = last_10_years.replace('Phoenix Coyotes', 'Arizona Coyotes')
last_10_years = last_10_years.replace('Atlanta Thrashers', 'Winnipeg Jets')
df[['Overall', 'Team', 'Player', 'Nat.', 'Pos', 'GP']].head(10)
teams = last_10_years.groupby('Team')['LinearDraftValue'].agg('sum').sort_values()

plt.figure(figsize=(10,7))
teams.plot(kind='barh', title='Linear Value of Overall Draft Pick Scoring by Team')
plt.show()
gp_by_draft_order = last_10_years.groupby('Overall')['GP'].agg('mean')

plt.figure(figsize=(10,7))
gp_by_draft_order.plot(title="Overall Draft Pick Value by Average Games Played")
plt.show()
last_10_years['DraftValue'] = last_10_years['Overall'].apply(lambda x: gp_by_draft_order.loc[x])

teams = last_10_years.groupby('Team')['DraftValue'].agg('sum').sort_values()

plt.figure(figsize=(10,7))
teams.plot(kind='barh', title='Value of Overall Draft Picks by Team')
plt.show()