! pip3 install pandas_alive
import pandas as pd

import pandas_alive

from glob import glob
path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]

df = pd.read_csv(path_ibov)

df.loc[:, "datetime"] =  pd.to_datetime(df.datetime)

df = df.set_index("datetime")

initial_date = "2000-01-01"

final_date = "2021-01-01"

df = df[initial_date:final_date]

df = df.reset_index()
pivoted_df = df.pivot(index='datetime',columns='ticker',values='volume').fillna(0)

pivoted_df = pivoted_df.resample("150d").mean()

del df

pivoted_df.plot_animated(filename='volume.gif',n_visible=10, period_fmt="%Y", title="Top 10 B3 Volume Leaders 2000-2020")
# ! ffmpeg -r 20 -i volume.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -crf 1 volume.mp4