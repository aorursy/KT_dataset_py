import pandas as pd
battles = pd.read_csv('/kaggle/input/game-of-thrones/battles.csv')
deaths = pd.read_csv('/kaggle/input/game-of-thrones/character-deaths.csv')
battles.head(7)
# Since there are no specifications for rows in () - default 5 rows will be displayed
deaths.tail()
battles.columns
battles.shape
deaths.shape[1]
deaths.shape[0]
deaths.iloc[7:11]
battles[battles.year == 299]
round(battles.major_death.mean(),2)
battles[ battles.year == 299 ].major_death.mean()
battles.year.max()
battles.year.sort_values(ascending=False).iloc[0]
battles.groupby('name').major_capture.mean()
battles.groupby('attacker_king').size()
battles.groupby('major_death').size().plot.pie()
deaths.groupby('Allegiances').size().plot.bar(x='Allegiance', y='Name')