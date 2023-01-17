# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

from tqdm import tqdm
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"

TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
!wc -l {TRAIN_PATH}
# datatypes for columns to use optimize memory usage

dtypes = {"pickup_datetime": "str", 

          "pickup_longitude": "float32", 

          "pickup_latitude": "float32", 

          "dropoff_longitude": "float32", 

          "dropoff_latitude": "float32", 

          "passenger_count": "uint8", 

          "fare_amount": "float32"}

cols = list(dtypes.keys())
chunk_size = 5_000_000 # 5 million rows per batch

# Read data in chunks and then concatenate them

df_list = []

for chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=dtypes, chunksize=chunk_size)):

    chunk["pickup_datetime"] = pd.to_datetime(chunk["pickup_datetime"].str.slice(0, 16), utc=True, format="%Y-%m-%d %H:%M")

    df_list.append(chunk)
# concat chunks in single dataframe

train = pd.concat(df_list)

del df_list
# Save into feather format formfaster reading next time

train.to_feather("nyc_taxi_fare_prediction.feather")