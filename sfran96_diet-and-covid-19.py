import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
import random
import scipy as sp
data = pd.read_csv("/kaggle/input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv")
data.info()
data.describe()
corr = data.loc[:, "Alcoholic Beverages":"Active"].drop("Undernourished", axis=1).corr()
plt.figure(figsize=(18,18))
ax = sb.heatmap(corr, cmap="RdBu", annot=True,)
ax.set_title("Correlation Heatmap of all food groups related variables + Coronavirus cases data")
ax.figure.show()
# Random seed
random.seed(7)

# Spain data
plt.figure(figsize=(8,8))
diet_europe = data.loc[[141, 74, 159], "Alcoholic Beverages":"Vegetal Products"].sort_index(axis=1)
colors = random.sample(list(cm.tab20b.colors) + list(cm.tab20c.colors),k = diet_europe.shape[1])
wedges, texts = plt.pie(diet_europe.mean().values.reshape(-1), colors=colors)
plt.legend(wedges, diet_europe.columns,
          title="Food type",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Average food group consumption by Italy, Spain and UK")
plt.show()
# South Korea data
plt.figure(figsize=(8,8))
diet_east_asia = data.loc[[76, 82, 121], "Alcoholic Beverages":"Vegetal Products"].sort_index(axis=1)
wedges, texts = plt.pie(diet_east_asia.mean().values.reshape(-1), colors=colors)
plt.legend(wedges, diet_east_asia.columns,
          title="Food type",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Average food group consumption by Japan, Philippines and South Korea")
plt.show()
# Differences in percentage on the types of foods consumed
plt.figure(figsize=(16,8))
plt.bar(diet_europe.columns, (diet_europe.mean().values-diet_east_asia.mean().values).reshape(-1), color="#ffd65c")
plt.xticks(rotation='vertical')
plt.title("Percentage (%) difference between Europe's and East Asia's diets")
plt.xlabel("Food group")
plt.ylabel("Percentage (%)")
plt.show()
data["recovered"] = (data["Recovered"]/data["Confirmed"])*100
data["deaths"] = (data["Deaths"]/data["Confirmed"])*100
data["active"] = (data["Active"]/data["Confirmed"])*100

data["AP Consumption"] = pd.cut(data["Animal Products"], [0, 10, 20, 30], labels=["low", "medium", "high"])
data.boxplot(column=["Obesity"], by="AP Consumption", figsize=(15,12))
data.boxplot(column=["Confirmed"], by="AP Consumption", figsize=(15,12))
plt.show()
fig, ax = plt.subplots()
data.plot(kind='scatter', x='Animal Products', y='Obesity', s=data["Confirmed"]*1000, c='deaths', cmap='plasma', figsize=(18,14), ax=ax) 
plt.show()