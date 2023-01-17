import os
import pandas as pd
# kernels let us navigate through the zipfile as if it were a directory
# os.chdir('../input/Data/ETFs/')
os.chdir('../input/Data/Stocks/')
# os.chdir('../../../input')
# os.listdir()[:5]
# the data is initially stored in many small csv files
os.listdir()[:5]
data = []
csvs = [x for x in os.listdir() if x.endswith('.txt')]
# trying to read a file of size zero will throw an error, so skip them
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.txt', '')
    data.append(df)
data = pd.concat(data, ignore_index=True)
data.reset_index(inplace=True, drop=True)

print(data.shape)
data.head()
os.chdir('../../../input')
os.listdir()[:5]
os.listdir("../")[:5]
data.to_csv("../working/usa_stocks_priceVol_11102017.csv.gz",index=False,compression="gzip")
os.chdir('Data/ETFs/')
os.listdir()[:4]
data = []
csvs = [x for x in os.listdir() if x.endswith('.txt')]
# trying to read a file of size zero will throw an error, so skip them
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.txt', '')
    data.append(df)
data = pd.concat(data, ignore_index=True)
data.reset_index(inplace=True, drop=True)

print(data.shape)

os.chdir('../../../input')
data.to_csv("../working/usa_ETF_priceVol_11102017.csv.gz",index=False,compression="gzip")
