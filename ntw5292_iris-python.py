#Import packages and data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("../input/Iris.csv")
#Print the head of the data set
print(iris.head())

#Print the shape of the data set
print(iris.shape)
#150 rows, 6 columns
#Plot data using matplotlib
iris.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")
#Seaborn jointplot to show scatterplot of data and univariate histograms simultaneously
sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=iris)
#Color the datapoints based on species
sns.FacetGrid(iris,hue="Species",size=7.5) \
   .map(plt.scatter,"SepalLengthCm","SepalWidthCm") \
   .add_legend() 
#Boxplot of Petal Length
sns.boxplot(x="Species",y="PetalLengthCm",data=iris)
#Striplot of Petal Length over the boxplot, use jitter so points don't fall in a straight line
ax = sns.boxplot(x="Species",y="PetalLengthCm",data=iris)
ax = sns.stripplot(x="Species",y="PetalLengthCm",data=iris,jitter=True,edgecolor="gray")
#kdeplot
sns.jointplot("SepalLengthCm","SepalWidthCm",kind="kde",data=iris)
#hexplot. change hexplot bin size using joint_kws, histogram bin size using marginal_kws
sns.jointplot("SepalLengthCm","SepalWidthCm",kind="hex",data=iris,joint_kws=dict(bins=10))
#The hexbin plots are the same for bin sizes >=5 since the data set is so small
#shaded kdeplot
setosa = iris.query("Species == 'Iris-setosa'")
virginica = iris.query("Species == 'Iris-virginica'")
versicolor = iris.query("Species == 'Iris-versicolor'")

f, ax = plt.subplots(figsize=(8,8))
ax.set_aspect("equal")

ax = sns.kdeplot(setosa.SepalWidthCm, setosa.SepalLengthCm, cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.SepalWidthCm, virginica.SepalLengthCm, cmap="Blues", shade=True, shade_lowest=False)
ax = sns.kdeplot(versicolor.SepalWidthCm, versicolor.SepalLengthCm, cmap="Greens", shade=True, shade_lowest=False)

#Shaded kdeplot may not be the best option for this particular data set since there is some overlap
#Is there a way to show overlap?
