import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_json('https://www.mohfw.gov.in/data/datanew.json')
df = df.iloc[:-1,:]
plt.subplots(figsize=(12,9))
sns.barplot(data=df,x='active',y='state_name')
plt.subplots(figsize=(12,9))
sns.barplot(data=df,x='positive',y='state_name')
df.set_index('state_name',drop= True).plot.bar(stacked=True,figsize=(12,10))
print(df.set_index('state_name',drop= True))
total_cases = df['positive'].sum()
print("total cases in india is {}".format(total_cases))