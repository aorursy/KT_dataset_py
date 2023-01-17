import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# https://www.kaggle.com/arjoonn/talkingdata-mobile-user-demographics/the-one-table/comments#132297

# we need to add dtype={'app_id': 'object', 'device_id': 'object'} as one

# of the arguments to pd.read_csv to read the IDs properly.

# example: 

#   app_ev = pd.read_csv('../input/app_events.csv', dtype={'app_id': 'object', 'device_id': 'object'})

# Read everything into memory

app_ev = pd.read_csv('../input/app_events.csv')

app_lab = pd.read_csv('../input/app_labels.csv')

ev = pd.read_csv('../input/events.csv')

la_cat = pd.read_csv('../input/label_categories.csv')

ph_br_dev_model = pd.read_csv('../input/phone_brand_device_model.csv')

gen_age_tr = pd.read_csv('../input/gender_age_train.csv')
# the ONE TABLE to rule them all

df = gen_age_tr.merge(ev, how='left', on='device_id')

df = df.merge(ph_br_dev_model, how='left', on='device_id')

df = df.merge(app_ev, how='left', on='event_id')

df = df.merge(app_lab, how='left', on='app_id')

df=  df.merge(la_cat, how='left', on='label_id')
df.info()
df.describe()
df.head()