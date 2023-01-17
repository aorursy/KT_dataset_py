import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
irisDF = pd.read_csv('../input/iris/Iris.csv')
print (irisDF.shape)
print (irisDF.columns)
irisDF.plot(kind='scatter',x="PetalLengthCm",y="PetalWidthCm")
plt.show()
irisDF.head()
sns.set_style("whitegrid")
sns.FacetGrid(irisDF,hue="Species",height=6)\
    .map(plt.scatter,"SepalLengthCm","SepalWidthCm")\
    .add_legend()
plt.show()
## 2-D Pair Plot == BIVARIATE

plt.close()
sns.set_style("whitegrid")
sns.pairplot(irisDF,hue="Species",height=3)
plt.plot()
plt.close()
sns.FacetGrid(irisDF,hue="Species",height=6)\
    .map(sns.distplot,'PetalLengthCm')\
    .add_legend()
plt.show()
#Distribution Plot on PetalWidthCm
sns.FacetGrid(irisDF,hue="Species",height=6)\
    .map(sns.distplot,"PetalWidthCm")\
    .add_legend()
plt.show()
#Distribution Plot on SepalLengthCm
sns.FacetGrid(irisDF,hue="Species",height=6)\
    .map(sns.distplot,"SepalLengthCm")\
    .add_legend()
plt.show()
#Distribution Plot on SepalWidthCm
sns.FacetGrid(irisDF,hue="Species",height=8)\
    .map(sns.distplot,"SepalWidthCm")\
    .add_legend()
plt.show()
#BOXPLOT ON IRIS DATA

sns.boxplot(x="Species",y="PetalLengthCm",data=irisDF)
plt.show()
#BOX PLOT ON IRIS DATA --- UNIVARIATE ON PETALWIDTHCM

sns.boxplot(x="Species",y="PetalWidthCm",data=irisDF)
plt.show()
#VIOLIN PLOT ON IRIS DATA === UNIVARIATE == PETALLENGTH

sns.violinplot(x="Species",y="PetalLengthCm",data=irisDF)
plt.show()


