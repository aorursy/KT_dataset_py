# import pandas

import pandas as pd
pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")
data = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

data.describe()
data.describe().transpose()