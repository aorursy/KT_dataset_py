import pandas as pd

import numpy as np
chipo = pd.read_table("../input/mydataset/chipotle.tsv",sep="\t")
chipo.head()
chipo['item_price'] = chipo['item_price'].str.replace('$', '')

chipo['item_price'] = chipo['item_price'].astype(float)
np.count_nonzero(chipo.item_price>10.00)
# Taking Min of item price so that only for 1 quantity the price is considered.

chipo1 = chipo.groupby(by ="item_name")[["item_price"]].min().reset_index()

chipo1
chipo.sort_values(by = "item_name",ascending=True).reset_index(drop=True)
xx = np.where(chipo.item_price == chipo.item_price.max(),chipo.quantity,0)

for i in xx:

    if i!=0:

        print("Quantity of the most expensive item ordered is : ",i)
temp = np.count_nonzero(chipo.item_name == "Veggie Salad Bowl")

print(temp,"times a Veggie Salad Bowl was ordered")
temp1 = np.count_nonzero((chipo.item_name == "Canned Soda") & (chipo.quantity > 1))

print(temp1,"times people orderd more than one Canned Soda")