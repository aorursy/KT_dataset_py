
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/plastic-top-20-rivers.csv')
df.head()
df['Plastic'] = df['Plastic mass input from rivers (tonnes)']
del df['Plastic mass input from rivers (tonnes)']
top = df[['Plastic','Entity']].sort_values(by='Plastic',ascending=False)
top_20 = top[:20]
top_20.reset_index(inplace=True)
top_20.drop('index',axis=1)
top_20.set_index('Entity',inplace=True)

plt.style.use('fivethirtyeight')
top_20['Plastic'].plot(kind='bar',figsize=(12,6))

plt.xlabel('Country')
plt.ylabel('Tonnes')
plt.title('World vs Asia vs Top 20 countries')
top_10_c = top_20[2:12]
plt.style.use('fivethirtyeight')
top_10_c['Plastic'].plot(kind='bar',figsize=(12,6))

plt.xlabel('Country')
plt.ylabel('Tonnes')
plt.title('Plastic pollution through rivers -Top 10 countries')