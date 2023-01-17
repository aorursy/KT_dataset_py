import numpy as np
import pandas as pd
pd.set_option('max_rows', 9)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
apps = pd.read_csv('../input/AppleStore.csv')
apps
price = apps.loc[:, ['track_name', 'price', 'size_bytes']]
price
plt.scatter(price.price, price.size_bytes / (1024 ** 2))

plt.title('Price vs. Size')
plt.xlabel('Price (USD)')
plt.ylabel('Size (MB)')

plt.show()
price_cleaned = price.loc[(price.price <= 50)]
price_cleaned
plt.scatter(price_cleaned.price, price_cleaned.size_bytes / (1024 ** 2))

plt.title('Price vs. Size (without outliers)')
plt.xlabel('Price (USD)')
plt.ylabel('Size (MB)')

plt.show()
plt.figure(figsize=(10, 7))
ax = sns.countplot(y=apps.prime_genre)
ax.set(xlabel='Number of apps', ylabel='Category')
plt.show()