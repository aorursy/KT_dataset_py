import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/train.csv')
from sklearn.cross_validation import train_test_split
df_sample = df.sample(frac=0.2, replace=False)
df_sample['dow'] = ((df_sample.time/1440)%7).astype(int)
df_sample['hour'] = ((df_sample.time/60)%24).astype(int)

df_sample_train, df_sample_test = train_test_split(df_sample, test_size=0.5, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5).fit(df_sample_train[['x','y','dow','hour']], df_sample_train['place_id'])
pred = rf.predict(df_sample_test[['x','y','dow','hour']])