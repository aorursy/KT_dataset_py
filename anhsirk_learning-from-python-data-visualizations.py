import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "white", color_codes = True)
iris = pd.read_csv("../input/Iris.csv")
iris.head()

iris["Species"].value_counts()
iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')
sns.jointplot(x='SepalLengthCm',y='PetalWidthCm',data=iris,size=5)
sns.FacetGrid(iris,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
ax = sns.boxplot(x="Species",y="PetalLengthCm",data=iris)
ax = sns.stripplot(x="Species",y="PetalLengthCm",data=iris,jitter=True,edgecolor='gray')
sns.violinplot(x='Species', y ='PetalLengthCm',data=iris,size=6)
sns.FacetGrid(iris, hue='Species',size=6).map(sns.kdeplot,'PetalLengthCm').add_legend()
sns.pairplot(iris.drop('Id',axis=1),hue='Species',size=3)
sns.pairplot(iris.drop('Id',axis=1),hue='Species',size=3,diag_kind='kde')
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop('Id',axis=1),'Species')
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop('Id',axis=1),'Species')
from pandas.tools.plotting import radviz
radviz(iris.drop('Id',axis=1),'Species')