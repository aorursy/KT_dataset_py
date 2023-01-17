import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.concat([pd.read_csv('../input/CallVoiceQuality_Data_2018_May.csv'),pd.read_csv('../input/CallVoiceQualityExperience-2018-April.csv')])
print(df.shape)
print(df.shape)
df.head()
df['Indoor_Outdoor_Travelling'].fillna(df['In Out Travelling'],inplace=True)
df.drop('In Out Travelling',axis-1,inplace=True)
df = df.loc[(df.Latitude != -1) & (df.Longitude != -1)]
print("Cleaned DF shape",df.shape)
## There are many duplicates , but this is OK given the data
df.drop_duplicates().shape
df.to_csv("IndiaCallQuality.csv",index=False)
df.drop_duplicates().to_csv("IndiaCallQuality_Deduplicated.csv.gz",index=False,compression="gzip")