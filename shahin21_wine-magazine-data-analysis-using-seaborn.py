import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data_path = "../input/wine-reviews/winemag-data_first150k.csv"

wine_data = pd.read_csv(data_path, index_col=0)

wine_data.head()
plt.figure(figsize=(50,8))

sns.countplot(wine_data["country"])
plt.figure(figsize=(16,6))

sns.countplot(wine_data["points"])
plt.figure(figsize=(60,8))

sns.countplot(wine_data[wine_data["price"]<=100]["price"])
plt.figure(figsize=(60,8))

sns.countplot(wine_data["province"].head(300))
plt.figure(figsize=(16,6))

sns.kdeplot(wine_data["points"])
plt.figure(figsize=(16,6))

sns.kdeplot(wine_data["price"])
plt.figure(figsize=(16,6))

sns.kdeplot(wine_data[wine_data["price"]<=100]["price"])
plt.figure(figsize=(16,6))

sns.kdeplot(wine_data.query("price <= 200").price)
sns.kdeplot(wine_data[wine_data['price']< 200].loc[:,['price', 'points']].dropna().sample(500))
plt.figure(figsize=(16,6))

sns.distplot(wine_data["points"], bins=40, kde=False)
ss2 = wine_data[['points', 'price']]

ss2.head()
sns.jointplot(x="price", y="points", data=wine_data[wine_data["price"]<200], height=8)
sns.jointplot(x="price", y="points", data=wine_data[wine_data["price"]<=15], kind="kde", height=8)
sns.jointplot(x="price", y="points", data=wine_data[wine_data["price"]<=100], kind="hex", gridsize=20, height=8)
plt.figure(figsize=(14,14))

sns.boxplot(x=wine_data["points"], y=wine_data["country"], data=wine_data)
plt.figure(figsize=(16,6))

dt = wine_data[wine_data.variety.isin(wine_data.variety.value_counts().head(10).index)]

sns.boxplot(x="variety", y="points", data=dt)
plt.figure(figsize=(50,6))

sns.boxenplot(x=wine_data["country"], y=wine_data["points"], data=wine_data)
plt.figure(figsize=(16,6))

dt = wine_data[wine_data.variety.isin(wine_data.variety.value_counts().head(10).index)]

sns.boxenplot(x="variety", y="points", data=dt)
plt.figure(figsize=(16,6))

dt = wine_data[wine_data.variety.isin(wine_data.variety.value_counts().head(10).index)]

sns.violinplot(x="variety", y="points", data=dt)