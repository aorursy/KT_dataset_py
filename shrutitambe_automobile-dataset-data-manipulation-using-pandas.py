import pandas as pd
auto = pd.read_csv("../input/automobiles/Automobile.csv")
auto.head()
auto.info()
auto["price"].mean()
auto[auto["price"] == auto["price"].max()]
auto[auto["price"] == auto["price"].min()]
auto[auto["horsepower"]>100].count()
auto[auto["body_style"] == "hatchback"].info()
auto["make"].value_counts().head(3)
auto[auto["price"]== 7099]["make"]
auto[auto["price"]>40000]
auto[(auto["body_style"]=="sedan") & (auto["price"]<7000)]