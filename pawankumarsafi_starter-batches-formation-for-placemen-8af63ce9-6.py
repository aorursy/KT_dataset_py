import pandas as pd
import xlrd
df=pd.read_excel("../input/batches/Batch_Formation_CSE_IT_MCA (1).xlsx")
df

df.head()
df.shape
df.tail()
df.head(2)
from matplotlib import pyplot as plt
import seaborn as sns
df['Stream'].value_counts() 

plt.figure(figsize=(7,7))
plt.bar(list(df['Stream'].value_counts()[0:5].keys()),list(df['Stream'].value_counts()[0:5]),color=["blue","green","pink","orange","red"])
plt.show() 

plt.hist(df['Stream'])
plt.show()

plt.pie(list(df['Stream'].value_counts()),labels=list(df['Stream'].value_counts().keys()),autopct='%0.1f%%')
plt.show()
df.describe()
df['Batch'].max()
df.index
df.loc[6]
df.sort_values('College Roll No.')
