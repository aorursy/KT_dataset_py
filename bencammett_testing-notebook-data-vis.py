import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
game_filepath = "../input/boardgamegeek-reviews/2019-05-02.csv"

game_data = pd.read_csv(game_filepath, index_col='Year')
print(list(game_data.columns))
sns.barplot(x=game_data.index, y=game_data['Average'])
plt.figure(figsize=(14,6))

sns.lineplot(data=game_data['Average'], label="Average")
plt.figure(figsize=(14,6))

sns.scatterplot(x=game_data['Average'], y=(game_data['Users rated']/max(game_data['Users rated'])))
# sns.jointplot(x=game_data['Average'], y=(game_data['Users rated']/max(game_data['Users rated'])), kind="kde")