import numpy as np
import pandas as pd
df = pd.read_csv("train.csv")
df['epoch'] = pd.to_datetime(df['epoch'], infer_datetime_format=True)
df['day'] = [epoch_dt.day for epoch_dt in df['epoch']]
jan_train = df[df['day'] < 25]
jan_test = df[df['day'] >= 25]
jan_train = jan_train.drop(['day'], axis=1)
answer_key = jan_test[['x','y','z','Vx','Vy','Vz']]
jan_test = jan_test[jan_test.columns.difference(['day','x','y','z','Vx','Vy','Vz'], sort=False)]
jan_train.to_csv('jan_train.csv', index=False)
jan_test.to_csv('jan_test.csv', index=False)
answer_key.to_csv('answer_key.csv', index=False)