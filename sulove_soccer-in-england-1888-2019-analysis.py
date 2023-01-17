import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
import seaborn as sns
df = pd.read_csv('../input/domestic-football-results-from-1888-to-2019/football_results.csv')
data = df[df['competition'] == 'england'] # Choosing Competition from England
data.head()
data.describe()
data [data['gh'] == 12]
data [data['ga'] == 9]
teams = data[['home','away']].stack().value_counts().rename_axis('Team').reset_index(name='counts')
teams
home = data[(data['gh'] > data['ga'])]
home = home['home'].value_counts().rename_axis('Team').reset_index(name='points')
home["points"] = 3 * home["points"]
home
away = data[(data['ga'] > data['gh'])]
away = away['away'].value_counts().rename_axis('Team').reset_index(name='points')
away["points"] = 3 * away["points"]
away
dagger = pd.concat([teams, away], axis=1)
dagger[['Team','Team']].stack().value_counts().tail(1) # We are doing tail(1) because there is only one team without away win
drawdata = data[(data['ga'] == data['gh'])]
drawdata
draw = drawdata[['home','away']].stack().value_counts().rename_axis('Team').reset_index(name='points')
draw # We don't need to multiply the matches by a number to get points because the drawn match = 1 point
home1 = home[home.Team != 'Glossop North End'].sort_values(by='Team', ascending=True).reset_index()
draw1 = draw[draw.Team != 'Glossop North End'].sort_values(by='Team', ascending=True).reset_index()
away1 = away.sort_values('Team').reset_index() # We don't have to remove 'Glossop North End' because it's not present here
#Therefore, all the above dataset is of similar size and nature and we can proceed with addition. Hope it works ! 
total = draw1['points'] + home1['points'] + away1['points']
aggregate = total.rename_axis('Team').reset_index(name='Points')
aggregate['Team'] = home1['Team']
aggregate.sort_values(by='Points', ascending = False).head(10).reset_index (drop = True)
top = aggregate.sort_values(by='Points', ascending = False).head(10)
sns.set(style="whitegrid")
plt.figure(figsize=(15,8))
sns.barplot(data = top, x = 'Points', y = 'Team', palette='Set2')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
plt.yticks( 
    fontweight='light',
    fontsize='16'  
)
plt.show()