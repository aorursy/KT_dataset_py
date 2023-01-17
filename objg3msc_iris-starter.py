import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
%matplotlib inline
print(check_output(["ls", "../input"]).decode("utf8"))
iris_df = pd.read_csv("../input/Iris.csv")
iris_df.describe()
iris_df.info()
iris_df.shape
iris_df.head()
# The different categories of Species
iris_df.Species.unique()
iris = iris_df.groupby('Species',as_index= False)["Id"].count()
iris
ax = iris_df[iris_df.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris_df[iris_df.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='white', label='versicolor',ax=ax)
iris_df[iris_df.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=ax)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Relationship between Sepal Length and Width")
sns.FacetGrid(iris_df, hue="Species", size=6) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()
plt.title("Relationship between Petal Length and Width")
cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')
petal = np.array(iris_df[["PetalLengthCm","PetalWidthCm"]])
sepal = np.array(iris_df[["SepalLengthCm","SepalWidthCm"]])

key = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
Y = iris_df['Species'].map(key)