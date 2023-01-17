import matplotlib.pyplot as plt

import pandas as pd
df=pd.read_csv("../input/company-sales/company_sales_data.csv")
#1)

df.head()
#2)

df.tail()
#3)

df.describe(include="all")
#4)

df.shape
#4)

df.size
#5)

type(df)
#5)

df.info()
plt.style.use("default")

plt.subplots(figsize=(5,3))

plt.plot(df.month_number,df.total_profit)

plt.grid()

plt.xlabel("Month Number",fontsize=10)

plt.ylabel("Total profit",fontsize=10)

plt.xticks([i for i in range(1,13)])

plt.title("Line plot showing total profit of all months",fontsize=15)

plt.show()
plt.style.use("default")

plt.subplots(figsize=(5,3))

plt.bar(df.month_number,df.bathingsoap)

plt.grid()

plt.xlabel("Month Number",fontsize=10)

plt.ylabel("Sale in No(bathing soap",fontsize=10)

plt.title("Bathing soap Sale Data",fontsize=15)

plt.xticks([i for i in range(1,13)])

plt.show()
# x is the total sales of each product in the year.i.e sum of products sold in 12 months.



plt.style.use("default")

labels = ['FaceCream', 'FaseWash', 'ToothPaste', 'Bathing soap', 'Shampoo', 'Moisturizer']

plt.pie(x=[34480,18515,69910,114010,25410,18515],autopct="%1.1f%%",labels=labels,explode=[0,0,0,0,0,0.2])

plt.show()