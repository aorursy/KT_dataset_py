from google.colab import files
files.upload()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/seasonality-values-d2b9a3.csv")
dataset.head()
dataset.id.unique()
select = dataset[dataset["id"]=="orange"]
select.head()
select["value"].plot()
dataset["year"], dataset["week"] = dataset.week_id.str.split('-').str
dataset[["week","year"]] = dataset[["week","year"]].astype(int)
dataset.head()
dataset.year.unique()
select = dataset[dataset.id=="orange"]
select.head()
sum(select[select.year==2004].value)
select2  = select.groupby("year").sum()["value"]
select2.head()
select2.plot()
