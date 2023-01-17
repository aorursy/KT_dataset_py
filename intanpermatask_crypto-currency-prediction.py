# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]



for ratio in ratios:

    dataset = f'/kaggle/input/crypto_data/crypto_data/{ratio}.csv'  # get the full path to the file.

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    

    # rename volume and close to include the ticker so we can still which close/volume is which:

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)



    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time

    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    

    if len(main_df)==0:  # if the dataframe is empty

        main_df = df  # then it's just the current df

    else:  # otherwise, join this data to the main one

        main_df = main_df.join(df)

        

print(main_df.head())