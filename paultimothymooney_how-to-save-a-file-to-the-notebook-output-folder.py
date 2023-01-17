import pandas as pd

PHD_STIPENDS = pd.read_csv('/kaggle/input/phd-stipends/csv') # load from notebook input

PHD_STIPENDS.to_csv('/kaggle/working/phd_stipends.csv',index=False) # save to notebook output

PHD_STIPENDS.head(10)