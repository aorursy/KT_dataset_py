import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
dataset= pd.read_csv("../input/Lemonade.csv")
dataset.head()
avg_sales = dataset["Sales"].mean()
print("Average Sales: ", avg_sales)
records_lower_avg = dataset[dataset["Sales"] < avg_sales]
print(records_lower_avg)
sales = dataset["Sales"].tolist()
#print(max(sales))
temperature = dataset["Temperature"].tolist()
#print(max(temperature))
holder = {'Sales' : sales,
         'Temperature' : temperature}

df = pd.DataFrame(holder)

df.plot(style=['o','rx'])
ax = plt.subplot()
ax.set_ylabel('Average Sales')
ed_avg_sales = dataset.groupby('Day').mean()['Sales']
print(ed_avg_sales)
ed_avg_sales.plot(kind='bar',figsize=(10,8), ax = ax)
