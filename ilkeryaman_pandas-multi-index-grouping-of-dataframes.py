import pandas as pd
from numpy.random import randn
outer_index = ["Group1", "Group1", "Group1", "Group2", "Group2", "Group2", "Group3", "Group3", "Group3"]

inner_index = ["Index1", "Index2", "Index3", "Index1", "Index2", "Index3", "Index1", "Index2", "Index3"]
hierarchy = list(zip(outer_index, inner_index))

hierarchy
hierarchy = pd.MultiIndex.from_tuples(hierarchy)

hierarchy
df = pd.DataFrame(randn(9, 3), hierarchy, columns=["Column1", "Column2", "Column3"])

df
df["Column1"]
df.loc["Group1"]
df.loc[["Group1", "Group2"]]
df.loc["Group1"].loc["Index1"]
df.loc["Group1"]["Column1"]
df.loc["Group1"].loc["Index1"]["Column1"]
df.loc["Group1"]["Column1"].loc["Index1"]
df.index.names # There is no index name so it can be confusing
df
df.index.names = ["Groups", "Indexes"]
df
df.loc["Group1"]
df.xs("Group1")
df.xs("Group2").xs("Index1").xs("Column1")
df.xs("Index1", level=1) # index level by index position
df.xs("Index1", level="Indexes") # index level by index name