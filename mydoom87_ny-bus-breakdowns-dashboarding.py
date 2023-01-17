import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import Dataset
dataset = pd.read_csv("../input/bus-breakdown-and-delays.csv")
#Get some info about the dataset and the structure
dataset.info()
dataset.head()
dataset['School_Year'].unique()
dataset[dataset['School_Year']=="2019-2020"]
#Delete the wrong data because one data occured in 2020 (Now it is 2018)
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d")

for i in range(0,len(dataset['Occurred_On'])):
    if dataset.iloc[i]['Occurred_On'] > now:
        df_new = dataset.drop(dataset.index[i])
df_new.info()
plt.figure(figsize=(15,6))
p = sns.countplot(x='Reason', data=df_new)
p.set_title('Breakdown and delay reasons')
for item in p.get_xticklabels():
    item.set_rotation(30)
sorted_year = sorted(df_new['School_Year'].unique())
p = sns.countplot(x='School_Year', data=df_new, order=sorted_year)
p.set_title('Breakdown and delay in different years')
plt.figure(figsize=(15,6))
p = sns.countplot(x='Reason', data=df_new, hue='School_Year')
p.set_title('Reasons in different years')
for item in p.get_xticklabels():
    item.set_rotation(30)
