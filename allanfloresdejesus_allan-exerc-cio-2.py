import pandas as pd



df = pd.read_csv("../input/dataviz-facens-20182-ex3/BlackFriday.csv")

df.head()
pd.DataFrame(df["Age"].value_counts())
import seaborn as sns



sns.violinplot(x=df.Age, y=df.Purchase)
import matplotlib.pyplot as plt



products = pd.DataFrame(df.Product_ID.value_counts().head(8))

fig_dims = (10, 5)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x=products.index, y=products.Product_ID, ax=ax)
top_5_ocupacoes = pd.DataFrame(df.Occupation.value_counts().head(5))

buyers_spendings = df[df.Occupation.isin(top_5_ocupacoes.index)]

sns.boxplot(data=buyers_spendings, x="Age", y="Purchase")

purchases_bigger_than_9000 = df[df.Purchase > 9000]

sns.violinplot(x = 'Marital_Status',

            y = 'Occupation',

            data = purchases_bigger_than_9000)