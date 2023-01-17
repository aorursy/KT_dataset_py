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
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df_complete=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_complete.head()
df_complete.dropna(axis=0)
df_selective=pd.DataFrame(df_complete, columns= ['State/UnionTerritory','Confirmed'])
df_selective.head()
df_confirmed=df_selective.groupby('State/UnionTerritory',as_index=False)['Confirmed'].sum()
df_confirmed.head()
df_confirmed
plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 number of confirmed cases w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_confirmed['State/UnionTerritory'], y=df_confirmed['Confirmed'])
plt.xticks(rotation=90)
plt.ylabel("Confirmed Cases Count")
df_cured=df_complete.groupby('State/UnionTerritory',as_index=False)['Cured'].sum()
df_cured
plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 Number of cured cases w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_cured['State/UnionTerritory'], y=df_cured['Cured'])
plt.xticks(rotation=90)
plt.ylabel("Cured Cases Count")
df_death=df_complete.groupby('State/UnionTerritory',as_index=False)['Deaths'].sum()
df_death
plt.figure(figsize=(10,6))
# title
plt.title("Covid-19 Number of deaths w.r.t state/union territory")
# Bar chart
sns.barplot(x=df_death['State/UnionTerritory'], y=df_death['Deaths'])
plt.xticks(rotation=90)
plt.ylabel("Number of deaths")