import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv("../input/indian_temp.csv")
print(df)
df1 = pd.read_excel("../input/indian_temp.xlsx")
df1
df2 = df1.iloc[0:20,0:13]
df2
df2.shape
df2
df2.describe()
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JAN',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='FEB',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAR',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='APR',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAY',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUN',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUL',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='AUG',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='SEP',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='OCT',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='NOV',data=df2)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='DEC',data=df2)
df3 = df1[['YEAR','JAN-FEB','MAR-MAY','JUN-SEP','OCT-DEC']]
df3 = df3.iloc[0:20,]
df3
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JAN-FEB',data=df3)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='MAR-MAY',data=df3)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='JUN-SEP',data=df3)
plt.figure(figsize=(20,8))
sb.barplot(x='YEAR',y='OCT-DEC',data=df3)
df1.corr()
df4 = df1[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']]
df4
df4 = df4.head(20)
df4
df5=df4.corr()
df5
plt.figure(figsize=(10,8))
sb.heatmap(df5,annot=True)
