import pandas as pd
Group=['X','X','Y','Y','Z','Z']
Data=[12,23,34,23,56,76]
df=pd.DataFrame({"Group":Group,"Data":Data})
df
Group1=[1,2,3,4]
Group2=[5,6,7,8]
Group3=[9,1,3,4]
df=pd.DataFrame({"Group1":Group1,"Group2":Group2,"Group3":Group3})
df
import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( y=df["sepal_length"] )
print(df.head())
import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["sepal_length"] )
print(df.head())
import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"] )
print(df.head())

import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot(data=df.iloc[:,0:2])
print(df.head())

import seaborn as sns
df = sns.load_dataset('tips')
sns.violinplot(x="day", y="total_bill", hue="smoker", data=df, palette="Pastel1")
print(df.head())
import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( y=df["species"], x=df["sepal_length"] )
print(df.head())

import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], linewidth=5)

sns.violinplot( x=df["species"], y=df["sepal_length"], width=0.3)
import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], palette="Blues")

import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], color="skyblue")
import seaborn as sns
df = sns.load_dataset('iris')
my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
sns.violinplot( x=df["species"], y=df["sepal_length"], palette=my_pal)
import seaborn as sns
df = sns.load_dataset('iris')
my_pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
sns.violinplot( x=df["species"], y=df["sepal_length"], palette=my_pal)

import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot(x='species', y='sepal_length', data=df, order=[ "versicolor", "virginica", "setosa"])

import seaborn as sns
df = sns.load_dataset('iris')
my_order = df.groupby(by=["species"])["sepal_length"].median().iloc[::-1].index
sns.violinplot(x='species', y='sepal_length', data=df, order=my_order)

