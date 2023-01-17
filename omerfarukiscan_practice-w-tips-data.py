import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
df =  pd.read_csv('../input/tipping/tips.csv')
df.head(3)
df.sort_values(by=['tip'], ascending=False).head()
df.sort_values(by=['total_bill'], ascending= False).head()
df.info()
sns.regplot(x="total_bill", y="tip", data=df, robust=True);
df.groupby(['sex']).mean()['tip']
df.groupby(['smoker']).mean()['tip']
sns.catplot(x="day", y="tip", data=df, height=6, kind="bar", palette="muted")
sns.catplot(x="time", y="tip", data=df, height=6, kind="bar", palette="muted")
plt.hist(df['tip'])
plt.xlabel('Dollars')
plt.ylabel('Number of tips')
plt.show()
plt.hist(df['total_bill'])
plt.xlabel('Dollars')
plt.ylabel('Number of bills')
plt.show()
labels = 'Male', 'Female'
sizes = [157, 87]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.show()
