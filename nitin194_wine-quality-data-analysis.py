your_local_path="../input/"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings; warnings.simplefilter('ignore')
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
wine = pd.read_csv(your_local_path+'winequality.csv')
wine.head()
print(wine.isnull().sum().sum())
printmd("**Above value shows that dataset is consistent with 0 missing values**")

print(wine.quality.describe())
sns.boxplot(wine.quality, linewidth=.5)
plt.show()
printmd("**1. It's evident from the above graph that the mean is somewhat dissolved with the median value which happens to be the boundry of the third quartile as well.**")
printmd("**2. Secondly graph shows that most of the quality values are either 5 or 6.**")

plt.hist(wine.quality)
plt.show()
printmd("**Above graph shows the actual distribution of quality values across the dataset and it validates that most of the wine quality values are either 5 or 6**")
q1=wine.quality.quantile(.25)
q3=wine.quality.quantile(.75)
iqr = q3-q1
outlier_low = wine.quality[wine.quality < (q1 - 1.5*iqr)].count()
outlier_high = outlier_low + wine.quality[wine.quality > (q3 + 1.5*iqr)].count()
printmd("**Number of outliers as per quality data is** : ")
print(outlier_high)
print(wine[['quality','fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].corr()['quality'].sort_values())

printmd("**Above chart shows that wine quality is strongly correlated with Alcohol(++), Volatile Actidity(--) and Sulphates(+)**")
#plt.figure(figsize=(100,10))
sns.pairplot(wine, x_vars=['volatile acidity','sulphates','alcohol'], y_vars=['quality'], kind="reg", size=5, aspect=1)
plt.show()
print(wine.groupby('quality')['alcohol'].mean())
printmd("\n **Average alcohol quantity tends to increase with quality** \n")
print(wine.groupby('quality')['sulphates'].mean())
printmd("\n **Average suplhate quantity tends to increase with quality** \n")
print(wine.groupby('quality')['volatile acidity'].mean())
printmd("\n **Average volatile acidity value tends to decrease with quality which depicts it can effect the quality negatively**")
bins = [0,3.3,6.6,10]
wine['cut'] = pd.cut(wine.quality, bins ,labels=['Low','Average','High'])
def get_stats(group):
    return {'min': group.min(), 'max':group.max(), 'mean':group.mean(), 'count':group.count()}

wine.quality.groupby(wine['cut']).apply(get_stats).unstack()
sns.distplot(wine[wine.cut == 'Low']['alcohol'], hist=False, kde_kws={"shade": True}, label="Low")
sns.distplot(wine[wine.cut == 'Average']['alcohol'], hist=False, kde_kws={"shade": True}, label="Average")
sns.distplot(wine[wine.cut == 'High']['alcohol'], hist=False, kde_kws={"shade": True}, label="High")
plt.show()
printmd("**Above graph indicates that alcohol content is generally more in high quality wine and vice versa**")
sns.distplot(wine[wine.cut == 'Low']['sulphates'], hist=False, kde_kws={"shade": True}, label="Low")
sns.distplot(wine[wine.cut == 'Average']['sulphates'], hist=False, kde_kws={"shade": True}, label="Average")
sns.distplot(wine[wine.cut == 'High']['sulphates'], hist=False, kde_kws={"shade": True}, label="High")
plt.show()
printmd("**Above graph indicates that sulphur content also tends to be more in high quality wine and vice versa**")
sns.distplot(wine[wine.cut == 'Low']['volatile acidity'], hist=False, kde_kws={"shade": True}, label="Low")
sns.distplot(wine[wine.cut == 'Average']['volatile acidity'], hist=False, kde_kws={"shade": True}, label="Average")
sns.distplot(wine[wine.cut == 'High']['volatile acidity'], hist=False, kde_kws={"shade": True}, label="High")
plt.show()
printmd("**Above graph indicates that volatile acidity content is generally lesser in high quality wine and vice versa**")
sns.heatmap(wine.corr())
plt.show()
printmd("**Above graph summarizes that the quality has a positive correlation with the alcohol and sulphates contents while negative correlation with the volatile acidity, chlorides, density and sulphur dioxide**")
