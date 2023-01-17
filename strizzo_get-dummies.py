import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
td=pd.read_csv("../input/train.csv")

td.describe()
pd.get_dummies(td).describe()