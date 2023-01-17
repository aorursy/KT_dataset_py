#Vitchayut Cheravinich 5410871
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/Lemonade.csv")

#Average Sales
average_sales = df['Sales'].sum()/df['Sales'].count()
print("Average Sales :" ,("%.2f" % average_sales))

#Sales lower than average
lta_sales = df[df.Sales < average_sales]
print(lta_sales)

#Scatter plot of sales and temperature
plt.xlabel("Sales")
plt.ylabel("Temperature")
plt.scatter (df['Sales'],df['Temperature'])
plt.show()

#Average Sales by days
day = df.groupby(['Day'])
group_day = day['Sales'].sum()/day['Sales'].count()

df_group = group_day
df_group.index = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
df_group.plot(kind='bar',stacked=True, figsize=(5,3))
from IPython.display import display
display(df_group)
