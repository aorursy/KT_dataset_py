import os

import pandas as pd
# kernels let us navigate through the zipfile as if it were a directory

os.chdir('../input/Data/ETFs/')
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
data.head()
oct10 = df[(df.Date == '2017-10-10')]

print(oct10.loc[oct10.Volume.idxmax()])