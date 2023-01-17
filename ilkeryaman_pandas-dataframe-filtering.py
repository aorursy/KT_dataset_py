import pandas as pd # data processing
from numpy.random import randn
df = pd.DataFrame(randn(4, 3), ["A", "B", "C", "D"], ["Column1", "Column2", "Column3"])

df
boolean_df = df > 0

df[boolean_df]
df["Column1"] > 0
df[df["Column1"] > 0]
df[(df["Column1"] > 0) & (df["Column2"] > 0)] # Get rows that values of Column1 and Column2 are bigger than 0.
df[(df["Column1"] < 0) & (df["Column2"] < 0)] # Get rows that values of Column1 and Column2 are lower than 0.
df[(df["Column1"] > 0) | (df["Column2"] > 0)]
df["Column4"] = pd.Series(randn(4), ["A", "B", "C", "D"])

df["Column5"] = randn(4)

df["Column6"] = ["new_value1", "new_value2", "new_value3", "new_value4"]

df
df.set_index("Column6") # set_index method has inplace parameter. Default value of inplace parameter is False. So it will not be set as permanent.
df
df.set_index("Column6", inplace=True)
df # index has changed from [A, B, C, D] to [new_value1, new_value2, new_value3, new_value4].
df.index.name # Name of index column
df.index.names # Dataframes can have multiple column as indexes. So there is another variable as names.