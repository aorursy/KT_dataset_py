import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



df = pd.read_csv("/kaggle/input/bostonhoustingmlnd/housing.csv")

df
df.query("MEDV>50000")
acceptable_home_prices = [501900, 470400, 249900, 728700]

df.query("MEDV in @acceptable_home_prices")
df.memory_usage()