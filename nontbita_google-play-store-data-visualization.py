import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
goog_play = pd.read_csv('../input/googleplaystore.csv')
goog_play.head()
goog_play.info()
goog_play.describe()
goog_play[goog_play['Rating'] == 19]
goog_play[10470:10475]
goog_play.iloc[10472,1:] = goog_play.iloc[10472,1:].shift(1)
goog_play[10470:10475]
goog_play.iloc[10472,1] = 'LIFESTYLE'
goog_play[10470:10475]
goog_play.describe()
goog_play.dtypes
goog_play['Rating'] = goog_play['Rating'].apply(pd.to_numeric, errors='coerce')
goog_play['Reviews'] = goog_play['Reviews'].apply(pd.to_numeric, errors='coerce')
goog_play.dtypes
#Histogram
goog_play["Rating"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Rating",figsize=(10,10))
plt.legend()
plt.xlabel("Rating")
plt.title("Rating Distribution")
plt.show()
#Histogram
goog_play["Reviews"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Reviews",figsize=(10,10))
plt.legend()
plt.xlabel("Reviews")
plt.title("Reviews Distribution")
plt.show()
