import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv("../input/heightweight.csv")
df.head()
y=df['HeightIn']
x=df['WeightLB']
df.head(10)
df.HeightIn.hist()
df['WeightLB'].hist()
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(x,y, color='r')
ax.set_ylabel('Boy', fontsize=10)
ax.set_xlabel('Kilo', fontsize=10)

df[df['HeightIn']>70]
df[df['gender']=='f'].hist()
df[df['gender']=='m'].hist()
df[df['WeightLB']>160]
plt.boxplot(df['WeightLB'])
