# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BreadBasket_DMS.csv")
df.head()

df.info()
#unique items bought
df["Item"].nunique()
#Count of individual items bought
df["Item"].value_counts()
#Graph of Number of items bought of particular category
plt.rc('figure', figsize=(20,5))
plt.bar(df["Item"].value_counts().index,df["Item"].value_counts())
plt.xticks(rotation=90)

#Hourly graph of transactions
hours = ["%.2d" % i for i in range(24)]
transaction_count = []
for j in range(24):
    
    if hours[j] == "23":
        time_df = df[(df["Time"] >= ""+hours[j]+":00:00") & (df["Time"] <= "00:00:00")]
    else:
        time_df = df[(df["Time"] >= ""+hours[j]+":00:00") & (df["Time"] <= ""+hours[j+1]+":00:00")]
    
    transaction_count.append(time_df["Item"].count())
    
plt.bar(hours,transaction_count)