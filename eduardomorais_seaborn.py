import seaborn as sns
df1 = sns.load_dataset("iris")

sns.pairplot(df1, hue="species")
df2 = sns.load_dataset('titanic')  

sns.pairplot(df1, hue="species")