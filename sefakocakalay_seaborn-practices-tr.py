import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
df.head()

sns.distplot(df.price);
sns.distplot(df.price,hist = False)
sns.kdeplot(df.price, shade = True)

cut_kategoriler = ["Fair","Good","Very Good","Premium","Ideal"]
df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler, ordered = True))
df.head()
sns.catplot(x ="cut", y ="price", data = df)
sns.barplot(x = "cut", y = "price" , hue = "color" , data = df)
