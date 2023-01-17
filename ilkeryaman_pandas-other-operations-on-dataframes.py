import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.DataFrame({
    "Column1": [1, 2, 3, 4, 5, 6],
    "Column2": [100, 100, 200, 300, 300, 100],
    "Column3": ["Ahmet", "Mehmet", "Ilker", "Hakan", "Mustafa", "Ali"]
})

df
df.head() # Returns first n rows (default = 5)
df.head(3) # Returns first 3 rows
df["Column2"].unique() # Retuns distinct values at Column2
df["Column2"].nunique() # Number of distinct values at Column2
df["Column2"].value_counts() # Occurances of values at Column2
df[(df["Column1"] >= 4) & (df["Column2"] == 300)] # Get indexes that value of Column1 is greater than and equal to 4 and value of Column2 equals 300.
def multiply_with(value, factor=3):
    return value * factor
df["Column2"].apply(multiply_with) # Multiply each value at Column2 with 3.
df["Column2"].apply(lambda x: x*2) # lambda functions can be applied also.
df["Column3"].apply(len)
df.drop("Column3", axis=1)
df.columns
df.index
len(df.index)
df.index.names
df.sort_values("Column2") # Sort ascending by Column2
df.sort_values("Column2", ascending=False) # Sort descending by Column2
df = pd.DataFrame({
    "Month": ["Mar", "Apr", "May", "Mar", "Apr", "May", "Mar", "Apr", "May"],
    "City": ["Ankara", "Ankara", "Ankara", "Istanbul", "Istanbul", "Istanbul", "Izmir", "Izmir", "Izmir"],
    "Moisture": [10, 25, 50, 21, 67, 80, 30, 70, 75]
})

df
df.pivot_table(index="Month", columns="City", values="Moisture")
df.pivot_table(index="City", columns="Month", values="Moisture")