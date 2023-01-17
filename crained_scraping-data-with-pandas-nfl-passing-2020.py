import pandas as pd
url = 'https://www.cbssports.com/nfl/stats/player/passing/nfl/regular/qualifiers/?sortdir=descending&sortcol=yds'
df_list = pd.read_html(url)
len(df_list)
df_list[0]