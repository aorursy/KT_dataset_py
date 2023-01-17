import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas import DataFrame

from pandas import Series

import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
retail=pd.read_csv('../input/retail_data.csv',encoding='unicode_escape')
df=DataFrame(retail)

df
df.describe()
df1=df[['Dealer Name','Status','Order QTY','City','Remarks','Pincode','Type of Project','Source','Order Val. (Rs. Lac)',' Order Recd\Lost ']]

df1

df3=df1[['Dealer Name']]



df4=df3.drop_duplicates()

df4
df4['Deals']=[36,20,63,15,45,20,16,14]

df4
df4['Lost']=[25,18,38,14,39,16,12,14]

df4
df4['Won']=df4['Deals']-df4['Lost']

df4
df4['Conversion Rate']=df4['Won']*100/(df4['Won']+df4['Lost'])

df4
plt.figure(figsize=(15,7))

plt.bar(df4['Dealer Name'],df4["Deals"])

plt.title("Dealer Name Vs Deals ")

plt.xlabel("Dealer Name")

plt.ylabel("Deals")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

plt.bar(df4["Dealer Name"],df4["Won"])

plt.title("Dealer Name Vs Won")

plt.xlabel("Dealer Name")

plt.ylabel("Won")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

plt.bar(df4["Dealer Name"],df4["Lost"])

plt.title("Dealer Name Vs Lost")

plt.xlabel("Dealer Name")

plt.ylabel("Lost")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

plt.bar(df4["Dealer Name"],df4["Conversion Rate"])

plt.title("Dealer Name Vs Conversion Rate")

plt.xlabel("Dealer Name")

plt.ylabel("Conversion Rate")

plt.xticks(rotation=90)

plt.show()
f, ax = plt.subplots(figsize=(15,7))

plt.bar(df4['Dealer Name'],df4["Deals"],label='Total Deals')

plt.bar(df4["Dealer Name"],df4["Won"],label='Deals Won')

plt.xticks(rotation=90)

plt.legend()

plt.show()
plt.figure(figsize=(15, 9))

sns.countplot(x='Dealer Name',hue='Order Received or Lost',data=df1)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15, 9))

sns.countplot(x='Dealer Name',hue='Remarks',data=df1)

plt.xticks(rotation=90)

plt.show()
df5=df1[['Dealer Name','Order Received or Lost','Remarks']]

df5.dropna()

df5.groupby(['Dealer Name','Order Received or Lost','Remarks']).sum()