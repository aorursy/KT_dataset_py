import pandas as pd

filename = '../input/ulabox_orders_with_categories_partials_2017.csv'
df = pd.read_csv(filename)
df.head()
import matplotlib.pyplot as plt
%matplotlib inline

plt.hist(df['hour'], bins=24)
plt.xlabel("Hour")
plt.ylabel("Orders")
babies = df[lambda x: x['Baby%']>50]
fresh = df[lambda x: x['Fresh%']>50]

plt.hist([fresh['hour'], babies['hour']], bins=24)
plt.xlabel("Hour")
plt.ylabel("Orders")