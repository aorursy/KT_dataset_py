import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.random import randn
arr = randn(3, 3)

arr
df = pd.DataFrame(data=arr, index=["A", "B", "C"], columns=["Column1", "Column2", "Column3"])

df # Here; Column1, Column2 and Column3 are series.
df["Column2"]
type(df["Column2"])
df.loc["A"] # Getting a row
df[["Column1", "Column3"]] # Gettint two columns
df["Column4"] = pd.Series(randn(3),index=["A", "B", "C"])

df
df["Column5"] = df["Column1"] + df["Column2"] + df["Column3"] # Make a new column named Column 5, that is total of Column 1, 2 and 3.

df
"""
There is a rule to drop column, row from DataFrame.
Axis information as (x = 0, y = 1) should be provided. Otherwise x is default.
Since inplace parameter's value is false as default, we won't see the result after drop.
"""
df.drop("Column5")
df.drop("Column5", axis=1)
df # Column5 is still here, because default value of inplace is False.
df.drop("Column5", axis=1, inplace=True)
df
df.iloc[0] # index A (Attention! it is iloc!)
df.iloc[1] # index B
df.iloc[2] # index C
df.loc["A"] # index A
df.loc["A", "Column2"] # Get value at row A and Column2.
df.loc[["A", "B"], ["Column1", "Column3"]] # Get values at Column 1 and Column3 at rows A and B.