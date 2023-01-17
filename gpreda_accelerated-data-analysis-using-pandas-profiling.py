import numpy as np 

import pandas as pd 

import os

import pandas_profiling
data_df = pd.read_csv(os.path.join('../input/housesalesprediction/','kc_house_data.csv'))
pandas_profiling.ProfileReport(data_df)