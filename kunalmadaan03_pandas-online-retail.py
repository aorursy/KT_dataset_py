import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
online_rt = pd.read_csv("../input/mydataset/Online_Retail.csv", encoding= 'unicode_escape')
online_rt.head()
mostQnty = online_rt.groupby(by="Country")[["Quantity"]].max().add_prefix("Max_")

mostQnty = mostQnty.sort_values(by="Max_Quantity",ascending=False).reset_index()
mostQnty.drop(mostQnty[mostQnty.Country == "United Kingdom"].index,inplace=True)

mostQnty = mostQnty.drop(mostQnty.index[10:]).reset_index(drop=True)
plt.figure(figsize=(12, 5))

sns.barplot(x="Country",y="Max_Quantity",data=mostQnty)

plt.xticks(rotation=25)

plt.show()
online_rt1 = online_rt[(online_rt.Quantity > 0)].reset_index(drop=True)
online_rt1
customers = online_rt.groupby(['CustomerID','Country']).sum()



# there is an outlier with negative price

customers = customers[customers.UnitPrice > 0]



# get the value of the index and put in the column Country

customers['Country'] = customers.index.get_level_values(1)



# top three countries

top_countries =  ['Netherlands', 'EIRE', 'Germany']



# filter the dataframe to just select ones in the top_countries

customers = customers[customers['Country'].isin(top_countries)]



# Graph Section 



g = sns.FacetGrid(customers, col="Country")



g.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)



# adds legend

g.add_legend()