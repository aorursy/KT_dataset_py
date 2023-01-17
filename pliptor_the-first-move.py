import pandas as pd

games = pd.read_csv("../input/games.csv", usecols=['moves','rated','white_id','white_rating'])
games.head()
print(games.shape)
games = games[games['rated']==True]
games.drop(['rated'],axis=1,inplace=True)
print(games.shape)
games['moves'] = games['moves'].apply(lambda x: x.split(' ')[0])
games.head()
games['moves'].value_counts(normalize=True)
games['white_rating'].describe()
games[games['white_rating']>1794]['moves'].value_counts(normalize=True)