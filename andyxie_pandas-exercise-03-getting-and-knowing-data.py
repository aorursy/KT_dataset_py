import pandas as pd
users = pd.read_csv("../input/u.user", sep="|", index_col="user_id")
users.head(25)
users.tail(10)
users.shape[0]
users.shape[1]
users.columns
users.index
users.dtypes
users["occupation"]
users["occupation"].value_counts().count()
users["occupation"].value_counts().sort_values(ascending=False).head()
users.describe()
users.describe(include="all")
users.occupation.describe()
users.age.mean()
users.age.value_counts().tail()