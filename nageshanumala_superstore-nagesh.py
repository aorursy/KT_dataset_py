import numpy as np

import pandas as pd
import os
path="../input"
os.chdir(path)
store=pd.read_csv("superstore_dataset2011-2015.csv",encoding="ISO-8859-1")
store.info()
store.shape[0] # How many rows
store.shape[1] # How many columns
store.index.values # Get the row names
store.columns.values # Get the columns names
store.sort_values("Profit", ascending=False)
store["Product Name"]
store.columns[0]
store.sort_values(["Profit","Discount"],ascending=[True,False])
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot("Category", data = store)
sns.countplot("Category",hue="Sub-Category" ,data = store)
sns.barplot(x="Category",y="Profit",hue="Sub-Category",data=store)