import pandas as pd



pd.read_parquet('/kaggle/input/binance-full-history/BNB-USDT.parquet').to_csv('BNB-USDT.csv')
# for filename in os.listdir('/kaggle/input/binance-full-history/'):

#     pairname = filename.split('.')[0]

#     pd.read_parquet(filename).to_csv(f'{pairname}.csv')