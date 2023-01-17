# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
frame1 = pd.read_csv("/kaggle/input/world-development-indicators/wdi-csv-zip-57-mb-/WDIData.csv")
frame1.describe()
frame1
frame1.columns =[column.replace(" ", "_") for column in frame1.columns] 

country_subset = frame1.query('Country_Code == "ARB" or Country_Code == "ZWE"')

topic_subset = country_subset.query('Indicator_Code == "EG.ELC.ACCS.ZS"')
topic_subset_cleaned = topic_subset.dropna(axis="columns")

topic_subset_cleaned2 = topic_subset_cleaned.drop(topic_subset_cleaned.columns[[0,1,2,3]], axis=1).transpose()

topic_subset_cleaned2.columns = ["Arab_World", "Zimbabwe"]

topic_subset_cleaned2
import seaborn as sns

sns.set(style="whitegrid")

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 6))



ax = sns.lineplot(data=topic_subset_cleaned2)