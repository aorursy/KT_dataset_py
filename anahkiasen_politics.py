import pandas as pd



data = pd.read_csv("../input/politicians/politicians.csv")

data.columns = ["name", "party", "company", "position", "paid"]

data.paid = [paid == "Yes"  for paid in data.paid]



data.head()
# Who pays for people?

data.groupby("company").paid.sum().sort_values(ascending=False)[:10].plot.barh()
# How many positions are paid?

data.paid.value_counts(normalize=True)
data.groupby("name").paid.sum().sort_values(ascending=False).plot.barh()