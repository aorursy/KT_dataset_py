# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, sys, glob



import json



# Any results you write to the current directory are saved as output.
tickdata = dict()

for root, dirs, filenames in os.walk('/kaggle/input'):

    for dirname in dirs:

        if dirname.startswith('histdata-forex-'):

            files_csv_zg = glob.glob(os.path.join(root, dirname,'*.csv.zg'))

            info=dict()

            missing = dict()

            with open(os.path.join(root, dirname, 'info.json')) as f:

              info = json.load(f)

            with open(os.path.join(root, dirname, 'missing.json')) as f:

              missing = json.load(f)

            provider, sectype, ticker = dirname.split('-',3)

            tickdata[str(dirname)] = {'provider':provider, 'ticker':ticker, 'sectype':sectype, 'details':info, 'count_of_missing_days':len(missing['days']), 'missing_days': missing['days'], 'files':files_csv_zg}
import tensorflow as tf
datasets = dict()

for key, values in tickdata.items():

    datasets[key]=tf.data.experimental.CsvDataset(values['files'], [tf.string,tf.float32,tf.float32],header=False, compression_type="GZIP",select_cols=[0,1,2])
input_ds = next(iter(datasets.values()))

input_ds.element_spec
for f in input_ds.take(5):

    print(f)
from datetime import datetime, timedelta
def conv_func(dt, bid, ask):

    txt = lambda t : t.numpy().decode('ascii')

    conv = lambda z : pd.Timestamp(datetime.strptime(z.numpy().decode('ascii'), '%Y%m%d %H%M%S%f')).to_datetime64()

    return tf.py_function(txt,[dt], tf.string), tf.py_function(conv, [dt], tf.float64), bid, ask,(bid+ask)/2, ask-bid
ts_data = dict()

for key, ds in datasets.items():

    ts_data[key] = ds.map(conv_func)

    
test_ds = next(iter(ts_data.values()))

test_ds.element_spec
for f in test_ds.take(5):

    print(f)
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

opendrive_usr = user_secrets.get_secret("OPENDRIVE_USERNAME")

opendrive_passwd = user_secrets.get_secret("OPENDRIVE_PASSWORD")
!pip install webdavclient3
import urllib3

from webdav3.client import Client

options = {

 'webdav_hostname': "https://webdav.opendrive.com",

 'webdav_login':    opendrive_usr,

 'webdav_password': opendrive_passwd

}

client = Client(options)

client.verify = False

urllib3.disable_warnings()
client.list()
if not client.check('ds_cache'):

    client.mkdir('ds_cache')
os.makedirs('/kaggle/working/cache', exist_ok=True)

for key, value in ts_data.items():

    cachedir = os.path.join('/kaggle/working/cache',key)

    os.makedirs(cachedir, exist_ok=True)

    value.cache(filename=cachedir)

    it = iter(value)

    list(next(it))

    # Uncomment follow to upload cache to opendrive

    # client.upload(remote_path="ds_cache/{}".format(key), local_path=cachedir)os