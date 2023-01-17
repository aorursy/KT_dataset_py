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

        file_location = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline





import matplotlib.dates as mdates



df = pd.read_csv(file_location)

df.head()
# in the title column there is a reason specified. It stands before the semicolon. I am creating a new column called reason to list all reasons 



df["reason"] = df["title"].apply(lambda title: title.split(":")[0])



df["reason"]
#breakdown of all calls by reason 

sns.countplot(x=df["reason"],palette = "viridis")
#timestamp column is a string object. I am using pandas built in function to convert it to DateTime object and create 3 new columns in df

df["timeStamp"]= pd.to_datetime(df["timeStamp"])





df["Month"] = df["timeStamp"].apply(lambda time: time.month)

df["Hour"] = df["timeStamp"].apply(lambda time:time.hour)

df["Day of Week"] = df["timeStamp"].apply(lambda time:time.dayofweek)



#df["Day of Week"] will show days as numbers. Hence next step is to create a dictionary and use map function to substitute numbers for day names



dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}



df["Day of Week"] = df["Day of Week"].map(dmap)
#now I am going to use seaborn to create countplots of day of the week column and month column,using reason column as a hue



sns.countplot(x = df["Day of Week"], hue = df["reason"], palette = "viridis")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)





sns.countplot(x = df["Month"], hue = df["reason"], palette = "viridis")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#finally im going to create a column Date, which is a date format of timeStamp column

df['Date']=df['timeStamp'].apply(lambda t: t.date())
# for my last graphs I will show the counnt for each call reason 



fig, axis = plt.subplots()

axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

df[df["reason"] == "Traffic"].groupby("Date").count()["twp"].plot(figsize=(15,10))

plt.title("Traffic")
fig, axis = plt.subplots()

axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

df[df["reason"] == "Fire"].groupby("Date").count()["twp"].plot(figsize=(15,10))

plt.title("Fire")
fig, axis = plt.subplots()

axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

df[df["reason"] == "EMS"].groupby("Date").count()["twp"].plot(figsize=(15,10))

plt.title("EMS")