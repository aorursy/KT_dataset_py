import pandas as pd
df = pd.read_csv('../input/gfp2017/GlobalFirePower_multiindex.csv', header=[0,1])
df.head()
df['Country Data'].head()
df['Country Data']['ISO3'].head()