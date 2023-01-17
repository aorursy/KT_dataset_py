import numpy as np
import pandas as pd
#importing data

df = pd.read_csv("../input/vizzdataset/skoruz nursery.csv")
df.head()
df.isnull().sum()
df.columns

df.info()
df_viz = df[["LotFrontage","MasVnrArea","GarageYrBlt","SalePrice"]]
df_viz.head()
df_viz.isnull().sum()
df
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,10)})
plt.figure(figsize=(15,10))
plt.plot(df_viz.LotFrontage)
sns.regplot(df_viz.LotFrontage, df_viz.SalePrice)
lot = df.LotFrontage
lot.describe()
lot.head()
lot.skew()
plt.hist(lot,bins = 40)
plt.show()
sns.distplot(lot, bins = 30)
snd_lot = lot.copy()
snd_lot.fillna(np.median(snd_lot))
snd_lot.isnull().sum()
snd_lot
lot.value_counts()
df_viz.info()
df_viz.describe()
lot_mean = np.mean(lot)
lot_mean
lot_std = np.std(lot)
lot_std
lower = lot_mean - lot_std*3
lower
upper = lot_mean + lot_std * 3
upper
outliers = [x for x in lot if x < lower or x > upper]
len(outliers)
outliers
outliers_rem = [x for x in lot if x not in outliers]
len(outliers_rem)
outliers_rem = pd.Series(outliers_rem)
outliers_rem.head()
outliers_rem.describe()
lot_clear = outliers_rem.copy()
lot_clear.isnull().sum()
plt.hist(lot_clear,bins = 40)
plt.show()
