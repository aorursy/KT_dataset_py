import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm 
df = pd.read_csv('/kaggle/input/insurance/test_Insurance.csv')
Insurance = pd.Series(df['#Insurance'], dtype='float') 
Insurance.index = pd.to_datetime(df['Month']) 
Insurance.plot()
#こちらにコードをお書きください
#こちらにコードをお書きください
#こちらにコードをお書きください
#こちらにコードをお書きください
#こちらにコードをお書きください
#こちらにコードをお書きください
#こちらにコードをお書きください