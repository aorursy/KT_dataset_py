import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
source_url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv'
df = pd.read_csv(source_url, index_col=0, parse_dates=[0])
df.head()