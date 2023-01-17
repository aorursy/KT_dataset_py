import pandas as pd

import pickle
%%time

PATH = '../input/data-science-bowl-2019'

train_df = pd.read_csv(PATH+'/train.csv')
%%time

train_df.to_pickle("./train.pkl")
%%time

train_pickle = pd.read_pickle('./train.pkl')