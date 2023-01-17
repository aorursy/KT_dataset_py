import numpy as np
import pandas as pd
calls = pd.read_csv('../input/911-calls/911.csv')
linhas, colunas = calls.shape
memoria = calls.memory_usage().sum()
print('linhas', linhas)
print('colunas', colunas)
print('memoria necessária', memoria)
print('\n----------\n')
print(calls.info())
calls['zip'].value_counts()
calls['twp'].value_counts()
len(calls['title'].unique().tolist())
reason = calls['title'].str.split(':', n = 1, expand = True)
calls['reason'] = reason[0]
calls
calls['reason'].value_counts()
print(calls['timeStamp'].dtypes)
calls['timeStamp'] = pd.to_datetime(calls['timeStamp'])
print(calls['timeStamp'].dtypes)
calls['month'] = calls['timeStamp'].dt.month
calls['dayofweek'] = calls['timeStamp'].dt.dayofweek
calls['hour'] = calls['timeStamp'].dt.hour
calls
calls['dayofweek'] = calls['dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
calls
calls.groupby(['month', 'reason']).size()
calls['timeStamp'].apply(lambda t: t.date()).value_counts().head(1)
calls.groupby(['dayofweek', 'hour']).size()['Fri'].idxmax()
calls.groupby(['dayofweek', 'reason']).size()['Sat']['Fire']
