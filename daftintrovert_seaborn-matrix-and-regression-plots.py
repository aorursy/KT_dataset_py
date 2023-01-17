import seaborn as sns

%matplotlib inline

tips = sns.load_dataset('tips')

flights = sns.load_dataset('flights')
tips.head()
flights.head()
tc = tips.corr()
tc
sns.heatmap(tc)
sns.heatmap(tc,annot = True,cmap = 'coolwarm')
flights.head()
ft = flights.pivot_table(index = 'month',columns = 'year',values = 'passengers')
sns.heatmap(ft)
sns.heatmap(ft,cmap = 'coolwarm',linewidth = 3,linecolor = 'white')
sns.clustermap(ft)
sns.clustermap(ft,cmap = 'coolwarm')
tips.head()
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,hue = 'smoker')
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,hue = 'sex',markers = ['o','v'])
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'sex')
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'sex',row = 'time')
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'day',hue = 'sex')
sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'day',hue = 'sex',aspect = 0.6,height = 8)
import seaborn as sns

%matplotlib inline

iris = sns.load_dataset('iris')

iris.head()
sns.pairplot(iris)
sns.pairplot(iris,hue = 'species')
#use of pair grid

import matplotlib.pyplot as plt
g = sns.PairGrid(iris)

g.map(plt.scatter)
g = sns.PairGrid(iris)

g.map_diag(sns.distplot)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)
tips.head()
g = sns.FacetGrid(data = tips, col = 'time',row = 'smoker')

g.map(sns.distplot,'total_bill')
g = sns.FacetGrid(data = tips, col = 'time',row = 'smoker')

g.map(sns.scatterplot,'total_bill','tip')
tips.head()
sns.set_style('darkgrid')

sns.countplot(x = 'sex',data = tips)

sns.despine()
sns.set_context('notebook')

sns.countplot(x = 'sex',data = tips)
sns.lmplot('total_bill','tip',data = tips,hue = 'sex',palette = 'inferno')
sns.lmplot('total_bill','tip',data = tips,hue = 'sex',palette = 'seismic')
#refer to matplotlib colormap section on google to get more insights on the changing patterns and getting differnt colormaps.