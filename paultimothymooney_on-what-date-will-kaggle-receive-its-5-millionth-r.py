import numpy as np

import pandas as pd

import datetime as dt

import os



def date_to_unix(year,month,day):

    date = pd.Series(dt.datetime(year,month,day))

    unix_epoch = date - dt.datetime(1970,1,1)

    unix_epoch = unix_epoch.dt.total_seconds().astype(str)

    return unix_epoch



def make_submission(your_guess,name_for_submission_file):

    data = {'id': [5000000], 

        'time': [your_guess]}

    df = pd.DataFrame(data)

    df.to_csv('/kaggle/working/'+name_for_submission_file+'.csv',index=False)

    return df.head()
my_guess = date_to_unix(2020,6,18)[0]

my_name_for_submission_file = 'mooneyp_submission_v1'

make_submission(my_guess, my_name_for_submission_file)