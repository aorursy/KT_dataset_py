import pandas as pd # data processing
dataset1 = {
    "A": ["A1", "A2", "A3", "A4"],
    "B": ["B1", "B2", "B3", "B4"],
    "C": ["C1", "C2", "C3", "C4"]
}

dataset2 = {
    "A": ["A5", "A6", "A7", "A8"],
    "B": ["B5", "B6", "B7", "B8"],
    "C": ["C5", "C6", "C7", "C8"]
}
df1 = pd.DataFrame(dataset1, index=[1, 2, 3, 4])

df2 = pd.DataFrame(dataset2, index=[5, 6, 7, 8])
df1
df2
pd.concat([df1, df2]) # Works based on Indexes, since axis = 0 (x)
pd.concat([df1, df2], axis=1) # Works based on Columns, since axis = 1 (y)
dataset1 = {
    "A": ["A1", "A2", "A3", "A4"],
    "B": ["B1", "B2", "B3", "B4"]
}

dataset2 = {
    "X": ["X1", "X2", "X3"],
    "Y": ["Y1", "Y2", "Y3"]
}
df1 = pd.DataFrame(dataset1, index=[1, 2, 3, 4])

df2 = pd.DataFrame(dataset2, index=[1, 2, 3])
df1
df2
df1.join(df2) # join works like left-join (works based on index)
df2.join(df1)
dataset1 = {
    "A": ["A1", "A2", "A3"],
    "B": ["B1", "B2", "B3"],
    "key": ["K1", "K2", "K3"]
}

dataset2 = {
    "X": ["X1", "X2", "X3", "X4"],
    "Y": ["Y1", "Y2", "Y3", "Y4"],
    "key": ["K1", "K2", "K3", "K4"]
}
df1 = pd.DataFrame(dataset1, index=[1, 2, 3])

df2 = pd.DataFrame(dataset2, index=[1, 2, 3, 4])
df1
df2
df1.merge(df2) # merges according to all common values (automatically) (works like an inner-join)
pd.merge(df1, df2, on=["key"]) # merges based on "key" column