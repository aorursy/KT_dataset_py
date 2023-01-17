import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
%matplotlib inline

pd.set_option('display.max_columns', 100)
df = pd.read_csv('../input/kds_full_monitoring.csv',parse_dates=["to_timestamp"],infer_datetime_format=True,low_memory=False)
print(df.shape)
df.head()
df.nunique()
df_entries = df.set_index("to_timestamp").resample("D")["user_id"].count()
print("total days:",df_entries.shape[0])
print("Days with visitors: :",(df_entries>0).sum())
df.drop(["user_id","domain_id"],axis=1,inplace=True)
df.head()
df.bot.describe()
# Note that we'll need to remove the org_name, if we want a model that can identify "stealth" bots!

df[df.bot==True].head()
df_entries.to_csv("kds_Daily_webtraffic.csv")
df.sample(frac=0.15).to_csv("kds_sample_webtraffic.csv.gz",compression="gzip")
