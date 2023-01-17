import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

df.head()
# The pattern in these plots is beatiful <3

plt.figure(figsize=(15,8))

df.snap_CA.plot()

plt.title('SNAP-CA')

plt.show()

plt.figure(figsize=(15,8))

df.snap_TX.plot()

plt.title('SNAP-TX')

plt.show()

plt.figure(figsize=(15,8))

df.snap_WI.plot()

plt.title('SNAP-WI')

plt.show()
#snap_CA

print("Years", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[0]).unique())

print("Months", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[1]).unique())

print("Days", df[df.snap_CA==1].date.apply(lambda x:x.split('-')[2]).unique())
#snap_TX

print("Years", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[0]).unique())

print("Months", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[1]).unique())

print("Days", df[df.snap_TX==1].date.apply(lambda x:x.split('-')[2]).unique())
#snap_WI

print("Years", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[0]).unique())

print("Months", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[1]).unique())

print("Days", df[df.snap_WI==1].date.apply(lambda x:x.split('-')[2]).unique())