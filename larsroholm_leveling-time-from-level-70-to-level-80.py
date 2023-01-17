# For handling data.

import pandas



# For plotting data.

%matplotlib inline

import seaborn

import matplotlib.pyplot as plot

seaborn.set(style = "darkgrid", palette = "husl")
# Load data.

data = pandas.read_csv("../input/wowah_data.csv")

data.columns = ["char", "level", "race", "charclass", "zone", "guild", "timestamp"]
# Create set of characters that reached level 80 and record the last timestamp at level 70 as well as the first timestamp at level 80.

last70 = data[data["level"] == 70].groupby("char", as_index=False).last()

ding80 = data[data["level"] == 80].groupby("char", as_index=False).first()

ding80.columns = ["char", "level", "race", "charclass", "zone", "guild", "ding80_timestamp"]

last70.columns = ["char", "level", "race", "charclass", "zone", "guild", "last70_timestamp"]

characters = pandas.merge(ding80[["char", "race", "charclass", "guild", "ding80_timestamp"]], last70[["char", "last70_timestamp"]], on="char")
# Parse timestamps.

characters["ding80_timestamp"] = characters["ding80_timestamp"].apply(pandas.to_datetime)

characters["last70_timestamp"] = characters["last70_timestamp"].apply(pandas.to_datetime)
# Create leveling time column.

characters["leveling_time"] = characters["ding80_timestamp"] - characters["last70_timestamp"]
# Remove high outliers in leveling time.

mean_leveling_time = characters["leveling_time"].mean()

std_leveling_time = characters["leveling_time"].std()

characters_no_slowpokes = characters[characters["leveling_time"] - mean_leveling_time <= 3 * std_leveling_time]
# Who was the top 10 fastest to hit 80?

characters[characters["leveling_time"].isin(characters["leveling_time"].nsmallest(10))].sort_values("leveling_time")
# Plot leveling time versus class.

seaborn.boxplot(x="charclass", y="leveling_time", data=characters_no_slowpokes)
# Plot leveling time versus guild.

plot.figure(figsize=(25,5))

seaborn.boxplot(x="guild", y="leveling_time", data=characters_no_slowpokes)
# Plot leveling time versus race.

seaborn.boxplot(x="race", y="leveling_time", data=characters_no_slowpokes)