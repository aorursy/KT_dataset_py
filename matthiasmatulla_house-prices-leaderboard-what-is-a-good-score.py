import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





house_lb = pd.read_csv('../input/hp-publicleaderboard/house-prices-advanced-regression-techniques-publicleaderboard.csv')

house_lb.info()
plt.figure(figsize=(16,6))

plt.xlabel("Score")

plt.xticks(np.arange(0, 1, step=0.025))

plt.ylabel("Frequency")

plt.hist(house_lb['Score'], range=(house_lb['Score'].min(),0.5) ,bins=100)

plt.show()
house_lb['Score'].value_counts().head(15).sort_index()
quantiles=[]

for q in np.arange(0, 21, step=1):

    quantiles.append(np.percentile(house_lb['Score'],q))

    

pd.set_option('display.max_columns', None)

df=pd.DataFrame({'Score':quantiles})

df.T
quantiles=[]

house_lb_=house_lb.loc[(house_lb['Score']!=0.00044)]

for q in np.arange(0, 21, step=1):

    quantiles.append(np.percentile(house_lb_['Score'],q))

    

pd.set_option('display.max_columns', None)

df=pd.DataFrame({'Score':quantiles})

df.T
plt.figure(figsize=(16,6))

plt.xlabel("Score")

plt.xticks(np.arange(0, 1, step=0.0005),rotation=90)

plt.ylabel("Frequency")

plt.hist(house_lb['Score'], range=(0.11,0.12385) ,bins=250)

plt.show()