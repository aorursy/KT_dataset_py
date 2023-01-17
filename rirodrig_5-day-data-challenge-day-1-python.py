import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#assign the read data to a variable called df 

df = pd.read_csv('../input/btcnCNY_1-min_data_2012-01-01_to_2017-05-31.csv')
df.describe().transpose()