import pandas as pd
users = pd.read_csv("../input/meta-kaggle/Users.csv")

users['Id'].nunique()