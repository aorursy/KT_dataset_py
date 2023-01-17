import pandas as pd
url = 'https://www.cbssports.com/nfl/stats/player/rushing/nfl/regular/qualifiers/'
df_list = pd.read_html(url)
len(df_list)
df_list[0]