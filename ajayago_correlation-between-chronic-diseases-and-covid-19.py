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
import matplotlib.pyplot as plt
chronic_dataset = pd.read_csv("/kaggle/input/uncover/us_cdc/us_cdc/500-cities-census-tract-level-data-gis-friendly-format-2019-release.csv")

confirmed_dataset = pd.read_csv("/kaggle/input/uncover/covid_tracking_project/covid-statistics-by-us-states-totals.csv")
confirmed_dataset
chronic_dataset.head()
chronic_dataset.columns
disease_data = chronic_dataset.filter(["stateabbr","cancer_crudeprev","geolocation"],axis=1)
count_disease=disease_data.groupby(disease_data["stateabbr"]).size()
disease_df=pd.DataFrame(count_disease).reset_index()

disease_df.columns=["state","count"]
plt.bar(disease_df["state"],disease_df["count"])

plt.xticks(rotation=90)

plt.figure(figsize=(80,80))

plt.show()
states_confirmed = confirmed_dataset.filter(["state","positive","negative"],axis=1)
states_confirmed
clubbed_df=pd.merge(disease_df,states_confirmed,on="state")
clubbed_df
clubbed_df.corr()
plt.bar(clubbed_df["state"],clubbed_df["count"])

plt.bar(clubbed_df["state"],clubbed_df["positive"],alpha=0.5)

plt.xticks(rotation=90)

plt.figure(figsize=(100,100))

plt.show()
sorted_clubbed=clubbed_df.sort_values(by=["count"])
plt.plot(sorted_clubbed["count"],sorted_clubbed["positive"])

plt.show()