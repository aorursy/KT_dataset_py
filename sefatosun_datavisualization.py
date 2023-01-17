import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
confirmed_df = pd.read_csv('/kaggle/input/covid19turkey/Covid19-Turkey.csv')
confirmed_df
plt.plot(confirmed_df['Daily Cases'])
plt.plot(confirmed_df['Total Cases'])

plt.title("Covid-19 Graph in TURKEY", fontsize=15)
plt.xlabel("Number of days", fontsize=15)
plt.ylabel("Cases", fontsize=15)
plt.legend(["Daily Cases","Total Cases"])
plt.plot(confirmed_df['Total Deaths'],color ='red')
plt.plot(confirmed_df['Total Recovered'], color ="green")
plt.title("Covid-19 Graph in TURKEY", fontsize=15)
plt.xlabel("Number of days", fontsize=15)
plt.ylabel("Cases", fontsize=15)
plt.legend(["Total Deaths","Total Recovered"])