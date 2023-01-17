#First of all , I will import seaborn librariy and load diamonds dataset.



import seaborn as sns

diamonds = sns.load_dataset('diamonds')

df = diamonds.copy()

df.head()
# We need some information about data.



df.info()
# And some statistic



df.describe().T
# Have more info about columns



df["cut"].value_counts()
df["color"].value_counts()
# ordinal definition



from pandas.api.types import CategoricalDtype
df.cut.head()
df.cut = df.cut.astype(CategoricalDtype(ordered = True))

df.dtypes
df.cut.head(1)
# that's why we need change the line.



cut_kategoriler = ["Fair", "Good", "Very Good", "Premium", "Ideal"]

df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler, ordered = True))

df.cut.head(1)
# You will se 2 different visualization technic here.



df["cut"].value_counts().plot.barh().set_title("Class Frequencies of Cut Variable");
sns.barplot(x = 'cut', y = df.cut.index, data = df);
# Crosswise cut with price and use catplot technic.



sns.catplot(x="cut", y="price", data=df);
# Crosswise more data now.



sns.barplot(x="cut",y="price",hue="color", data=df)
df.groupby(["cut","color"])["price"].mean()
# Use displot for explore more information.



sns.distplot(df.price, kde=False);
df["price"].describe()
# bins = 100 help us to determine the definition of range.



sns.distplot(df.price,bins = 100, kde = False);
# bins = 1000 show more detailed data.



sns.distplot(df.price,bins = 1000, kde = False);
# Of course, you can make better decisions with less details.



sns.distplot(df.price,bins = 10, kde = False);
# Sometimes a line can help us read the data.



sns.distplot(df.price);
# Or you can use just a line.



sns.distplot(df.price, hist= False);
#Anyway types can be enhanced.



sns.kdeplot(df.price, shade=True);
# More columns help for better decision.



(sns

 .FacetGrid(df,

             hue = "cut",

             height= 5,

             xlim =(0,10000))

 .map(sns.kdeplot,"price",shade=True)

 .add_legend()

);
#Of course, different graphic types can also help us.



sns.catplot(x="cut", y="price",hue="color",kind="point", data =df);