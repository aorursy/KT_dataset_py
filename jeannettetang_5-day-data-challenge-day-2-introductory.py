

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv("../input/cereal.csv")

data.describe()
import seaborn as sns # visualization library



calories = data["calories"]

sns.distplot(calories, kde = False).set_title("Calories")