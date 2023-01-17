import pandas as pd 
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/african_conflicts.csv",encoding="latin1",low_memory=False)

print(df.columns)
df.head()
COLS_KEEP = ['ADMIN1','ADMIN2', 'ADMIN3', 'COUNTRY',
       'EVENT_DATE', 'EVENT_TYPE',
       'FATALITIES',  'GWNO',  'LATITUDE', 'LOCATION', 'LONGITUDE' ]

# FATALITIES is a LEAK if used, but we will leave it in in case we want to builda model to predict fatalities, given the event

df = df[COLS_KEEP]
df.head()
df.EVENT_TYPE.value_counts()
df.EVENT_TYPE.replace({"Violence Against Civilians": "Violence against civilians","Strategic Development":"Strategic development","Strategic development ":"Strategic development" },inplace=True)
df.EVENT_TYPE.value_counts()
df.EVENT_TYPE.isna().sum()
df.to_csv("african_conflicts_events.csv.gz",index=False,compression="gzip")
