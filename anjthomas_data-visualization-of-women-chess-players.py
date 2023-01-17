import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

chess_file_path = "../input/top-women-chess-players/top_women_chess_players_aug_2020.csv"

#load data
chess_data = pd.read_csv(chess_file_path)
chess_data.head()
#check size of data
chess_data.shape
chess_data['Title'].value_counts()
chess_data.isna().sum()
#Popluate NaN values with 0
chess_data.fillna(0,inplace=True)
#lets add Age column

current_year = datetime.datetime.now().year #get current year
chess_data['Age'] = current_year - chess_data['Year_of_birth'].astype(int)
chess_data['Inactive_flag'].replace({'wi':'inactive'}, inplace=True)
chess_data['Inactive_flag'].replace({0:'active'}, inplace=True)
chess_data.head()
player_rating = chess_data.groupby('Name')['Standard_Rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(10,6))
plt.title("Mean Standard Ratings")

sns.barplot(data=player_rating, y='Name', x='Standard_Rating', palette='Set1')
player_rating = chess_data.groupby('Name')['Blitz_rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(10,6))
plt.title("Mean Blitz Ratings")

sns.barplot(data=player_rating, y='Name', x='Blitz_rating')
player_rating = chess_data.groupby('Name')['Rapid_rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(10,6))
plt.title("Mean Rapid Ratings")

sns.barplot(data=player_rating, y='Name', x='Rapid_rating', palette='Set2')
age_df = chess_data[chess_data['Age']!=0]
age_df = age_df.groupby('Name').mean().sort_values(by = ['Standard_Rating','Rapid_rating', 'Blitz_rating'],ascending=False).reset_index().head(10)

plt.figure(figsize=(10,6))
plt.title("Age of top 10 Players")

sns.barplot(data = age_df, y='Name', x='Age', palette="Paired")
df = chess_data[chess_data['Title']!=0][chess_data['Year_of_birth']!=0]
plt.figure(figsize=(10,7))
sns.swarmplot(x=df['Title'], y=df['Age'], hue=df['Inactive_flag'])
fed_df = chess_data[chess_data['Federation']!=0][chess_data['Year_of_birth']!=0]
fed_df = fed_df[fed_df['Federation'].isin(fed_df['Federation'].value_counts().head(10).to_frame().index)]


sns.catplot(data=fed_df, x='Federation', y='Age', hue='Inactive_flag', kind="boxen", height=5, aspect=4).set(title="Age plot of Federations with Most Players")
rating_df = chess_data[chess_data['Standard_Rating']!=0][chess_data['Blitz_rating']!=0][chess_data['Rapid_rating']!=0]
plt.figure(figsize=(10,8))
sns.set(style="white", palette="bright", color_codes=True)
sns.kdeplot(data = rating_df['Standard_Rating'], shade=True, label = "Standard Rating")
sns.kdeplot(data = rating_df['Blitz_rating'], shade=True, label = "Blitz Rating")
sns.kdeplot(data = rating_df['Rapid_rating'], shade=True, label = "Rapid Rating")

plt.legend()
fed_rating = chess_data[chess_data['Federation'] != 0]
fed_std_rating = fed_rating.groupby('Federation')['Standard_Rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(12,8))
plt.title("Mean Standard Ratings")

sns.barplot(data=fed_std_rating, x='Federation', y='Standard_Rating')
fed_blitz_rating = fed_rating.groupby('Federation')['Blitz_rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(12,8))
plt.title("Mean Blitz Ratings")

sns.barplot(data=fed_blitz_rating, x='Federation', y='Blitz_rating', palette="coolwarm")
fed_rapid_rating = fed_rating.groupby('Federation')['Rapid_rating'].mean().sort_values(ascending=False).reset_index().head(10)

plt.figure(figsize=(12,8))
plt.title("Mean Rapid Ratings")

sns.barplot(data=fed_rapid_rating, x='Federation', y='Rapid_rating', palette="GnBu_d")
blitz_rapid = chess_data[chess_data['Blitz_rating']!=0][chess_data['Rapid_rating']!=0]

plt.figure(figsize=(10,8))

sns.set(palette = "Set1")
sns.jointplot( data=blitz_rapid, x='Blitz_rating', y='Rapid_rating', kind="kde")
rating_df = chess_data.groupby('Federation').mean().sort_values(by = ['Standard_Rating', 'Rapid_rating', 'Blitz_rating'], ascending=False)
rating_df = rating_df[(rating_df!=0).all(1)] # dropping all rows with atleast one zero value
rating_df = rating_df.drop(['Fide id', 'Year_of_birth','Age'], axis=1).head(10)

plt.figure(figsize=(15,10))
plt.title("Top ratings of first 10 Federations")
sns.heatmap(data = rating_df, annot=True, fmt='g', cmap='Blues')
top_df = chess_data.groupby('Name').mean().sort_values(by = ['Standard_Rating', 'Rapid_rating', 'Blitz_rating'], ascending=False)
top_df = top_df[(top_df!=0).all(1)] # dropping all rows with atleast one zero value
top_df = top_df.drop(['Fide id', 'Year_of_birth','Age'], axis=1).head(10)

plt.figure(figsize=(15,10))
plt.title("Top ratings of first 10 Players")
sns.heatmap(data = top_df, annot=True, fmt='g', cmap=sns.light_palette("green"))
title_rating = chess_data[chess_data['Title']!=0][chess_data['Standard_Rating']!=0]

sns.catplot(data=title_rating, x='Title', y='Standard_Rating', kind='violin', palette='Set2', aspect=2)
sns.catplot(data=title_rating, x='Title', y='Blitz_rating', kind='violin', palette='husl', aspect=2)
sns.catplot(data=title_rating, x='Title', y='Rapid_rating', kind='violin', palette='Set3', aspect=2)
fed_rating = chess_data[chess_data['Federation']!=0][chess_data['Standard_Rating']!=0]
fed_rating = fed_rating[fed_rating['Federation'].isin(fed_rating['Federation'].value_counts().head(10).to_frame().index)]
sns.catplot(data=fed_rating, x='Federation', y='Standard_Rating', kind='box', palette='pastel', aspect=2)
fed_rating = chess_data[chess_data['Federation']!=0][chess_data['Blitz_rating']!=0]
fed_rating = fed_rating[fed_rating['Federation'].isin(fed_rating['Federation'].value_counts().head(10).to_frame().index)]
sns.catplot(data=fed_rating, x='Federation', y='Blitz_rating', kind='box', palette='Set1', aspect=2)
fed_rating = chess_data[chess_data['Federation']!=0][chess_data['Rapid_rating']!=0]
fed_rating = fed_rating[fed_rating['Federation'].isin(fed_rating['Federation'].value_counts().head(10).to_frame().index)]
sns.catplot(data=fed_rating, x='Federation', y='Rapid_rating', kind='box', palette='Paired', aspect=2)