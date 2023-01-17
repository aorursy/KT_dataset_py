import pandas as pd
url = 'https://www.cbssports.com/nba/stats/player/scoring/nba/postseason/all-pos/qualifiers/?sortdir=descending&sortcol=ppg'
df_list = pd.read_html(url)
len(df_list)
df_list[0]