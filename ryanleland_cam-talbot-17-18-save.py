import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the data set into pandas
goalies = json.loads(open('../input/goalies.json', 'r').read())['data']
df = pd.DataFrame(goalies)

# Filter to goalie we're interested in, and pull out the relevant data
df = df[df['playerName'] == 'Cam Talbot']
df = df[['gameDate', 'timeOnIce', 'shotsAgainst', 'saves', 'savePctg']]

df.set_index('gameDate')
df.sort_values('gameDate')
print(df.describe())
df['savePctg'].hist(bins=20)
plt.show()
df['savePctg'].rolling(5, min_periods=5).mean().plot()
plt.show()