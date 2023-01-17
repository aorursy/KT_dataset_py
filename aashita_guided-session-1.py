import pandas as pd
# Uncomment the below two lines only if using Google Colab

# from google.colab import files

# uploaded = files.upload()

df = pd.read_csv('../input/archive.csv')
# df
df.shape
df.head()
df.tail(10)
df.columns
df[['Punxsutawney Phil', 'February Average Temperature (Pennsylvania)']].head()
df[df["Punxsutawney Phil"]=="No Record"]
    
df.loc[:3, ['Year', 'Punxsutawney Phil']]
# df.iloc[:3, ['Year', 'Punxsutawney Phil']] # This will give an error

# Comment the above line of code and uncomment the below one

df.iloc[:3, [0, 1]] 
df.iloc[10]
df.iloc[0:len(df):10]