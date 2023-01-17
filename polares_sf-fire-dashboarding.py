# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

fire_calls = pd.read_csv('../input/fire-department-calls-for-service.csv')
fire_calls['Call Date'] = pd.to_datetime(fire_calls['Call Date'])
fire_calls['Watch Date'] = pd.to_datetime(fire_calls['Watch Date'])
fire_calls['Received DtTm'] = pd.to_datetime(fire_calls['Received DtTm'])
fire_calls['Entry DtTm'] = pd.to_datetime(fire_calls['Entry DtTm'])
fire_calls['Dispatch DtTm'] = pd.to_datetime(fire_calls['Dispatch DtTm'])
fire_calls['Response DtTm'] = pd.to_datetime(fire_calls['Response DtTm'])
fire_calls['On Scene DtTm'] = pd.to_datetime(fire_calls['On Scene DtTm'])
fire_calls['Transport DtTm'] = pd.to_datetime(fire_calls['Transport DtTm'])
fire_calls['Hospital DtTm'] = pd.to_datetime(fire_calls['Hospital DtTm'])
fire_calls['Available DtTm'] = pd.to_datetime(fire_calls['Available DtTm'])
fire_calls.info()
fire_calls.head()
now = datetime.now()
week_ago = now - timedelta(days=7)
weekly_calls = fire_calls[fire_calls['Call Date'] > week_ago]
weekly_calls.shape[0]
weekly_calls['Call Type'].value_counts().plot.bar(figsize = (12,8))
plt.title(f'{week_ago.day}.{week_ago.month}.{week_ago.year} - {now.day}.{now.month}.{now.year} Incident type dist.');
log_call_type = pd.DataFrame(np.log(weekly_calls['Call Type'].value_counts()))
log_call_type.plot.bar(figsize = (12,8))
plt.bar(range(len(log_call_type)), log_call_type["Call Type"], color=plt.cm.Paired(np.arange(len(log_call_type))))
plt.title(f'{week_ago.day}.{week_ago.month}.{week_ago.year} - {now.day}.{now.month}.{now.year} Incident type logarithmic dist.');
weekly_calls[pd.to_numeric(weekly_calls['Original Priority'], errors='coerce') < weekly_calls['Final Priority']].shape[0]