import pandas as pd
data = pd.read_csv("../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv")
data.head(10)
data.describe()