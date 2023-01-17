import pandas as pd

data = pd.read_csv("../input/archive.csv")

data
data.describe().transpose()