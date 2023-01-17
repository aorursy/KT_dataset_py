# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/bot-detection/ibm_data.csv")
df
print(df.operating_sys.isnull().sum())
df.operating_sys.fillna("NotGiven",inplace=True)
print(df.operating_sys.unique())
def shortenos(x):
    #print(x)
    if "microsoft" in x.lower().split("_")[0]:
        x="MICROSOFT PC"
        return x
    elif "windowsphone" in x.lower().split("_")[0]:
        x="WINDOWS MOBILE"
        return x
    elif "windowsmobile" in x.lower().split("_")[0]:
        x="WINDOWS MOBILE"
        return x
    elif "macintosh" in x.lower().split("_")[0]:
        x="MACOS PC"
        return x
    elif "ios" in x.lower().split("_")[0]:
        x="IOS PHONE"
        return x
    elif "android" in x.lower().split("_")[0]:
        x="ANDROID"
        return x
    elif "linux" in x.lower().split("_")[0]:
        x="LINUX"
        return x
    elif x.lower()=="notgiven":
        x="NotGiven"
        return x
    else:
        x="OTHER"
        return x
df["os"]=df.operating_sys.apply(shortenos)
  
os_df=df.groupby(["os"]).sum().reset_index()
os_df

import altair as alt
import altair_render_script
alt.data_transformers.disable_max_rows()
base=alt.Chart(os_df).mark_bar().encode(
x="os",
y="VISIT",
tooltip=["VISIT"]
)

base2=alt.Chart(os_df).mark_bar().encode(
x="os",
y="VIEWS",tooltip=["VIEWS"]
)
alt.hconcat(base,base2)
df.user_agent.dropna(inplace=True)
!pip install device_detector
from device_detector import SoftwareDetector

def parse_family(x):
    
    return SoftwareDetector(x).parse().client_name()
def parse_os(x):
  
    return SoftwareDetector(x).parse().os_name()


    
sample_df=df[:400000]
sample_df.user_agent.dropna(inplace=True)

x=sample_df["user_agent"].apply(parse_family)
sample_df["user_browser"]=x
sample_df.user_agent.dropna(inplace=True)
sample_df["user_os"]=sample_df.user_agent.apply(parse_os)

sample_df
from device_detector import DeviceDetector
def parse_is_bot(x):
    return DeviceDetector(x).parse().is_bot()
sample_df.user_agent.dropna(inplace=True)
sample_df["is_bot"]=sample_df["user_agent"].apply(parse_is_bot)
def replace_empty_user(x):
    if x=="":
       return "Unknown"
    else:
        return x
sample_df["user_os"]=sample_df["user_os"].apply(replace_empty_user)
user_os_df=sample_df.groupby(["user_os","is_bot"]).count().reset_index()
user_os_df.rename(columns={"Unnamed: 0": "count"},inplace=True)
user_os_df

base=alt.Chart(user_os_df[user_os_df.is_bot==True]).mark_bar().encode(
x="user_os",
y="count",
    tooltip=["count"]
).properties(title="BOTS")

base1=alt.Chart(user_os_df[user_os_df.is_bot==False]).mark_bar().encode(
x="user_os",
y="count",
     tooltip=["count"]
).properties(title="Not_BOTS")
alt.hconcat(base,base1)


browser_df=sample_df.groupby(["user_browser","is_bot"]).count().reset_index()
browser_df.rename(columns={"Unnamed: 0": "count"},inplace=True)

base=alt.Chart(browser_df[browser_df.is_bot==True]).mark_bar().encode(
x="user_browser",
y="count",
    tooltip=["count"]
).properties(title="BOTS")

base1=alt.Chart(browser_df[browser_df.is_bot==False]).mark_bar().encode(
x="user_browser",
y="count",
     tooltip=["count"]
).properties(title="Not_BOTS")
alt.hconcat(base,base1)
sample_df.to_csv("bots_firsthalf.csv",index=False)
sample_df2=df[400000:]
sample_df2.user_agent.dropna(inplace=True)
x=sample_df2["user_agent"].apply(parse_family)
sample_df2["user_browser"]=x
sample_df2.user_agent.dropna(inplace=True)
sample_df2["user_os"]=sample_df2.user_agent.apply(parse_os)

sample_df2.user_agent.dropna(inplace=True)
sample_df2["is_bot"]=sample_df2["user_agent"].apply(parse_is_bot)
sample_df.to_csv("bots_secondhalf.csv",index=False)
the_big_df=pd.concat([sample_df,sample_df2])
the_big_df.to_csv("bots_full.csv",index=False)

