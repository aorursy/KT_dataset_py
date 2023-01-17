import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

df.head()
# remove the unnamed column

df.drop(['Unnamed: 0'], axis=1,inplace=True)
# Replace -1 or -1.0 or '-1' to NaN

df=df.replace(-1,np.nan)

df=df.replace(-1.0,np.nan)

df=df.replace('-1',np.nan)
# drop null values

df = df.dropna()
# top industries

i = df['Industry'].value_counts(dropna=False)

i.head(20)
sns.barplot(x=df['Industry'].value_counts()[0:9],y=df['Industry'].value_counts()[0:9].index)
# top companies hiring data analysts

c = df['Company Name'].value_counts(dropna=False)

c.head(20)
sns.barplot(x=df['Company Name'].value_counts()[0:9],y=df['Company Name'].value_counts()[0:9].index)
# top locations  

df["Location"].value_counts().sort_values(ascending=False).head(20).plot.bar(color='b')

plt.title("Top Locations for Data Analyst Jobs",fontsize=20)

plt.xlabel("Locations",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()