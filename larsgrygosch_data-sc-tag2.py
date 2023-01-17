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
# Schauen wir uns zunächst die README an für einen ersten Einblick
with open('/kaggle/input/sleepstudypilot/README.md') as f:
    for line in f.readlines():
        print(line)
# Laden der Datei
df = pd.read_csv("/kaggle/input/sleepstudypilot/SleepStudyData.csv")
# Datensatz anschauen (die ersten 5 Einträge)
df.head()
# Oder alternativ die letzten 5 Einträge
df.tail()
# Informationen
df.info()
df.Hours
#df['Hours']  # Alternativ
df.head()
df.iloc[1][4]
df.loc[0]['Hours']
df[df.Hours < 6].head()
df[df.Breakfast == "Yes"].head()
df[df.Breakfast == "Yes"].count()
df[df.Hours.isnull()]
df.describe()
df_enough = df.groupby(['Enough'])
df_enough.describe()
# Lösung 1

df_breakfast = df[df.Breakfast == 'Yes']
df_breakfast.describe()
# Lösung 2
df.groupby(['Breakfast']).describe()
# Cleaning
df_copy = df
df_copy[df_copy.Hours.isnull()]
breakfast_mean = round(df_copy[df.Breakfast == 'Yes'].Hours.mean(), 2)
print(breakfast_mean)
# Vorher
print(df_copy.loc[65]["Hours"], df_copy.loc[91]["Hours"]) 

df_copy = df_copy.fillna(breakfast_mean)
# Nachher
print(df_copy.loc[65]["Hours"], df_copy.loc[91]["Hours"]) 
df_copy.info()
# Correltaion

df_copy.corr()
df_gr = df_copy.groupby(["Breakfast", "PhoneReach"])
df_gr.describe()
df_gr.corr()


