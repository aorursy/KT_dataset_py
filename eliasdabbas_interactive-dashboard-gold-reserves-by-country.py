from IPython.display import IFrame

IFrame(src='https://www.dashboardom.com/gold-reserves', width='100%', height=600)
import pandas as pd

gold = pd.read_csv('../input/gold_quarterly_reserves_ounces.csv')

gold['Time Period'] = [pd.Period(p) for p in gold['Time Period']]

gold.head()
gold.dtypes
# download URL: https://data.imf.org/?sk=388DFA60-1D26-4ADE-B505-A05A558D9A42&sId=1479329132316

# International Financial Statistics dataset

# ifs = pd.read_csv('path/to/file.csv')

# metric_name = 'International Reserves, Official Reserve Assets, Gold (Including Gold Deposits and, If Appropriate, Gold Swapped), Volume in Millions of Fine Troy Ounces , Fine Troy Ounces'

# gold_quarterly_oz = ifs[(ifs['Indicator Name']==gold_res_oz) & (ifs['Time Period'].astype('str').str.contains('Q'))]