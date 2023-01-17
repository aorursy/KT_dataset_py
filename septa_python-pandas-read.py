# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd

import numpy as np

import datetime



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Remove comments, if you want to see files.

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Routine to parse dates

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



# Read data from both files



d=pd.read_csv("../input/trainView.csv",

    header=0,names=['train_id','status','next_station','service','dest','lon',

                    'lat','source','track_change','track','date','timeStamp0',

                    'timeStamp1','seconds'],

    dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,

    'lon':str,'lat':str,'source':str,'track_change':str,'track':str,'date':str,

    'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime,'seconds':str}, 

     parse_dates=['timeStamp0','timeStamp1'],date_parser=dateparse)







o=pd.read_csv("../input/otp.csv",

    header=0,names=['train_id','direction','origin','next_station','date','status',

                    'timeStamp'],

    dtype={'train_id':str,'direction':str,'origine':str,'next_station':str,

                           'date':str,'status':str,'timeStamp':datetime.datetime}, 

    parse_dates=['timeStamp'],date_parser=dateparse)
# trainView.csv

d.head()
# otp.csv

o.head()