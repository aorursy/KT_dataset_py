import pandas as pd
url = 'https://www.msn.com/en-au/sport/football/epl/player-stats'
df_list = pd.read_html(url)
len(df_list)
df_list[0]