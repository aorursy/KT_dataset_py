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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("../input/911-calls-dataset-capstone-project/911.csv")

df.info()
df["zip"].value_counts().head()
df["twp"].value_counts().head()
# Take a look at the title column, How many unique title codes
df["title"].nunique()

df["Reason"] =df["title"].apply(lambda reason :reason.split(":")[0])
df["Reason"]
sns.countplot(x="Reason",data=df,palette="viridis")
df["timeStamp"].dtypes
df["timeStamp"]= pd.to_datetime(df["timeStamp"])
df["timeStamp"].dtypes
df["Hour"] =df["timeStamp"].apply(lambda time :time.hour)
df["Month"] =df["timeStamp"].apply(lambda time : time.month)
df["Day of Week"] =df["timeStamp"].apply(lambda time : time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df["Day og Week"] = df["Day of Week"].map(dmap)
sns.countplot(x ="Day of Week", data=df,hue="Reason",palette="viridis")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x ="Month", data=df,hue="Reason",palette="viridis")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
bymonth =df.groupby("Month").count()
bymonth.head()
bymonth.twp.plot()
sns.lmplot(x="Month",y="twp",data=bymonth.reset_index())
df["Date"] =df["timeStamp"].apply(lambda t :t.date())
df["Date"]
df.groupby("Date").count()["twp"].plot()
plt.tight_layout()
df[df["Reason"]=="EMS"].groupby("Date")["twp"].count().plot()
plt.title("EMS")
df[df["Reason"]=="Fire"].groupby("Date")["twp"].count().plot()
plt.title("Fire")
df[df["Reason"]=="Traffic"].groupby("Date")["twp"].count().plot()
plt.title("Traffic")
dayHour =df.groupby(by=["Day of Week","Hour"]).count()["Reason"].unstack()
dayHour.head()
sns.heatmap(dayHour,cmap ="viridis")
sns.clustermap(dayHour,cmap="viridis")
dayMonth =df.groupby(by=["Day of Week","Month"]).count()["Reason"].unstack()
dayMonth.head()
sns.heatmap(dayMonth,cmap="viridis")
sns.clustermap(dayMonth,cmap="viridis")
