import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df_events = pd.read_csv("../input/events.csv", encoding='utf-8')
df_events.head()
df_gender_age = pd.read_csv('../input/gender_age_train.csv', encoding='utf-8')
df_gender_age.head()
df_phone_data = pd.read_csv('../input/phone_brand_device_model.csv', encoding='utf-8')
df_phone_data.head()
# merge the first two tables
df_merged = pd.merge(df_events, df_gender_age, on='device_id', how='left')
df_merged.head()
# merge the next table
df_merged = pd.merge(df_merged, df_phone_data, on='device_id', how='left')
df_merged.head()
singled_out = df_merged[df_merged['device_id'] == 29182687948017175]

singled_out['longitude']