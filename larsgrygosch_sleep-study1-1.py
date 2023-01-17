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
df = pd.read_csv("/kaggle/input/sleepstudypilot/SleepStudyData.csv")
df
df.info()
df.describe()
df["Hours"].hist()
df.corr()
df.groupby("Tired").describe()
df.groupby(["Breakfast", "Tired"]).describe()
df.info()
df[df["Hours"].isnull()]
df1 = df[["Enough", "Hours", "Tired", "Breakfast"]]
df1
mean_breakfast = df[df["Breakfast"] == "Yes"].mean()["Hours"].round(1)
df1 = df1.fillna(mean_breakfast)
df1
df1.info()
# Wieviele der Teilnehmer haben mehr als 6 Stunden Schlaf gehabt

# Wieviele der Teilnehmer haben mehr als 6h Schlaf gehabt und sich trozdem mit 4- 5 bewerten bei tired
df_filter = df1[df1["Hours"]>6]
df_filter[df_filter["Tired"] >= 4].describe()
df[df["Hours"] > 6].groupby("PhoneTime").corr()
df[df.duplicated()]
