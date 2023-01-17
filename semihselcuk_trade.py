# Installation

!pip install fxcmpy 

!pip install python-socketio
import fxcmpy

import pandas as pd

import datetime as dt

from pylab import plt
# Start connection

# con = fxcmpy.fxcmpy(config_file=r'D:\\Coding\\Projects\\PyTrade\\src\\fxcm.cfg')

TOKEN = 'e5b98178914e49c5e7bddcbc0fc799d9c37062db'

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error',

                    server='demo', log_file='log.txt')

# Get 4 hours of DATA

data = con.get_candles('USD/TRY', period='H4') 

data
# Data Visualisation

plt.style.use('seaborn')

%matplotlib inline

data.plot(figsize=(10, 6), lw=0.8);