import pandas as pd 

import os

print ("These are the files you can load:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import gc

gc.collect()

print ("LSTM Dataset")

df = pd.read_csv('/kaggle/input/m5-accuracy-preprocessed-datasets/lstm_df.csv')

gc.collect()

df.info()
gc.collect()

print ("Dimensionality Reduction Dataset")

df = pd.read_csv('/kaggle/input/m5-accuracy-preprocessed-datasets/dimred_df.csv')

gc.collect()

df.info()