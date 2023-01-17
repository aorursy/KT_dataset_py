# This is an example url of a competition. Note that you download the "leaderboard" html page

url = 'https://www.kaggle.com/c/firstoffour/leaderboard'
import pandas as pd # Let pandas do its magic!
data = pd.read_html('/kaggle/input/kaggle.html')
df = data[0]

df.head()
final_rankings = df[['Team Name','Score']]
final_rankings.head()