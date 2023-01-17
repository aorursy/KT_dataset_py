import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!ls ../input
df = pd.read_csv("../input/temp.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Temp'] = df['Temp'].str.replace(',','.').astype('float')
df.plot.line(x='Date', y='Temp', figsize=(16,9))