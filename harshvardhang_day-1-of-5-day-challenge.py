import pandas as pd;

pd.set_option('display.float_format', lambda x: '%.3f' % x) 





df = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv')

df.dtypes

df.describe()