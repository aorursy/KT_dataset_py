import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

    

print("Setup Complete")
import pandas as pd

despesas_diarias = pd.read_csv("../input/diarias/diarias-formated.csv", index_col="credor")

despesas_diarias.head()
## Todos os veradores

plt.figure(figsize=(14,6))

sns.set(style="darkgrid")



sns.lineplot(data=despesas_diarias, palette="tab10", linewidth=2.5)



plt.xlabel("credor")


### Por ano



plt.figure(figsize=(14,6))



plt.title("Gastos com diaria")



sns.lineplot(data=despesas_diarias['2019'], label="2019", palette="tab10", linewidth=2.5)



# Add label for horizontal axis

plt.xlabel("credor")
### Por ano



plt.figure(figsize=(14,6))



plt.title("Gastos com diaria")



sns.lineplot(data=despesas_diarias['2018'], label="2018", palette="tab10", linewidth=2.5)



# Add label for horizontal axis

plt.xlabel("credor")
### Por ano



plt.figure(figsize=(14,6))



plt.title("Gastos com diaria")



sns.lineplot(data=despesas_diarias['2017'], label="2017", palette="tab10", linewidth=2.5)



# Add label for horizontal axis

plt.xlabel("credor")