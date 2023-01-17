print('hello')
import pandas as pd

df = pd.read_csv(r"..//input//jeopardy-dataset//jeopardy.csv")
print(df.head(20))
print(df.tail(20))
print(df.info())
print(df.shape)
print(list(df.columns))
columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']

df.columns = columns
df = df[pd.notnull(df['Answer'])]

values = df['Value'] != 'None'

df = df[values]
df['Air Date'] = pd.to_datetime(df['Air Date'],format='%Y-%m-%d')

df.set_index('Air Date',inplace=True)
# categories names and round names 

categories = df['Category'].unique()

round = df['Round'].unique()



# rounds 

round_1 = df[df['Round'] == "Jeopardy!"]

round_2 = df[df['Round'] == "Double Jeopardy!"]
df['Value'] = df['Value'].str.strip('$')

df['Value'] = df['Value'].str.replace(',','')
df['Value'] = pd.to_numeric(df['Value'])
df['Year'] = df.index.year

df['Month'] = df.index.month
jeopardy = df