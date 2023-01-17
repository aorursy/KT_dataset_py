import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
games_data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
games_data.head()
drop_row_index = games_data[games_data['Year'] > 2015].index
games_data = games_data.drop(drop_row_index)
games_data.shape
games_data['Genre'].value_counts()
plt.figure(figsize=(15, 10))
sns.countplot(x="Genre", data=games_data, order=games_data['Genre'].value_counts().index)
plt.xticks(rotation=90)
games_data['Year'].value_counts()
plt.figure(figsize=(15, 10))
sns.countplot(x='Year', data=games_data, order=games_data.groupby(by=['Year'])['Name'].count().sort_values(ascending=False).index)
plt.xticks(rotation=90)
plt.figure(figsize=(30, 10))
sns.countplot(x="Year", data=games_data, hue='Genre', order=games_data.Year.value_counts().iloc[:5].index)
plt.xticks(size=16, rotation=90)
