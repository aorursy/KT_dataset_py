import pandas as pd

df = pd.read_csv('/kaggle/input/CAvideos.csv', delimiter=',')

df.dataframeName = 'USvideos.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
views = df["trending_date"].value_counts()

views.value_counts() 
artists = df["channel_title"].value_counts() 

artists.head(5)