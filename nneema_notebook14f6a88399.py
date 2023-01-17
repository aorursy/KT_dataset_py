%matplotlib inline



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/Salaries.csv")
data.describe()
filtered = data[data["BasePay"] != "Not Provided"]

# gb = data.groupby([data["Year"],data["JobTitle"]])

filtered.count

#filtered.groupby(["Year", "JobTitle"])["BasePay"].mean()
type("ss")