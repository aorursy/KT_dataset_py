import pandas as pd

import os
os.listdir('../input/')
df = pd.read_csv("../input/UserInfo.tsv")

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t")

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",header=0)

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",header=1) # it makes first data row as header

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",header=[1,2,3,4,5])

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",header=0)

df.head()
df = pd.read_csv("../input/UserInfo.tsv",delimiter='\t') #alias for sep

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",encoding="utf-8")

df.head()
df.head(10)
df.tail(10)
len(df)
df = pd.read_csv("../input/UserInfo.tsv",sep="\t",nrows=20, encoding="utf-8")

df
len(df)
df = pd.read_csv("../input/UserInfo.tsv",sep="\t", skiprows=100)

df.head()
len(df)
df = pd.read_csv("../input/UserInfo.tsv",sep="\t", skiprows=100, header=None)

df.head()
df = pd.read_csv("../input/UserInfo.tsv",sep="\t", usecols=["UserID","UserAgentID","UserAgentOSID","UserDeviceID","UserAgentFamilyID"])

df.head()
len(df)
df = pd.read_csv("../input/UserInfo.tsv",sep="\t", skipfooter=100)

df.head()
len(df)