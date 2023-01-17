import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn

%matplotlib inline
# Data provided by https://www.superdatascience.com/
df = pd.read_csv("../input/office-supplies-data/OfficeSupplies.csv")
df.head()
df_units = df[["Rep", "Units"]]
df_units.head()
df_units.groupby("Rep").sum()
df_units.groupby("Rep").sum().head()
rep_plot = df_units.groupby("Rep").sum().plot(kind='bar')
rep_plot.set_xlabel("Rep")
rep_plot.set_ylabel("Units")
df["Total Price"] = df["Units"] * df["Unit Price"]
df.head()
df.sort_values("Total Price", ascending=False).head()
df.groupby("Rep").sum().sort_values("Total Price", ascending=False).plot(kind='bar')
df_items = df[["Item", "Total Price"]]
df_items.groupby("Item").sum().plot(kind="bar")
df_region = df[["Region", "Total Price"]]
df_region.groupby("Region").sum().plot(kind="bar")
group = df.groupby(["Region","Rep"]).sum()
total_price = group["Total Price"].groupby(level=0, group_keys=False)

gtp = total_price.nlargest(5)
ax = gtp.plot(kind="bar")

#draw lines and titles
count = gtp.groupby("Region").count()
cs = np.cumsum(count)
for i in range(len(count)):
    title = count.index.values[i]
    ax.axvline(cs[i]-.5, lw=0.8, color="k")
    ax.text(cs[i]-(count[i]+1)/2., 1.02, title, ha="center",
            transform=ax.get_xaxis_transform())

# shorten xticklabels
ax.set_xticklabels([l.get_text().split(", ")[1][:-1] for l in ax.get_xticklabels()])