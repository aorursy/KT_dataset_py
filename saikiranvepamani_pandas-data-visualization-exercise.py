import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('../input/salaries/df3')
%matplotlib inline
df3.info()
df3.head()
df3.plot.scatter(x='a',y='b',c='red',s=50,figsize=(10,5))
df3['a'].plot.hist()
plt.style.use('ggplot')
df3['a'].plot.hist(alpha=0.3,bins=30)
df3[['a','b']].plot.box()
df3['d'].plot.kde()
df3['d'].plot.kde(lw=3,ls='--')
df3.loc[0:30].plot.area(alpha=0.4)
df3.loc[0:30].plot.area(alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))