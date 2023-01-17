import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
%matplotlib inline
df = pd.DataFrame(np.random.rand(500,4),columns='a b c d'.split())
df.head()
df.info()
df.plot.scatter(x='a',y='b',figsize=(12,3),s=80,c='orange',edgecolor='red')
df['a'].plot.hist(edgecolor='violet',color='purple')
plt.style.use('ggplot')
df['a'].plot.hist(bins=25,edgecolor='pink',layout='squeeze',alpha=0.5)
df[['a','b']].plot.box()
df['d'].plot.kde(lw=5,ls='--')
df.iloc[0:30].plot.area(alpha=0.5,cmap='coolwarm')
df.iloc[0:30].plot.area(alpha=0.5,cmap='coolwarm').legend(loc='center left', bbox_to_anchor=(1,0.5))
# OOP Style
f = plt.figure()
df.iloc[0:30].plot.area(alpha=0.4,ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
