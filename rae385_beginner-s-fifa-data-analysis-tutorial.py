import pandas as pd
player_dataset = pd.read_csv("../input/data.csv", index_col = 0) #save dataset as pd DataFrame object
player_dataset.head(5)
player_dataset.Nationality.head(5)
nationality_count = player_dataset.sort_values(by='Nationality').Nationality.value_counts()

nationality_count.head(10)
brazilian_players = player_dataset.loc[(player_dataset.Nationality == "Brazil")]

brazilian_players.head(5)