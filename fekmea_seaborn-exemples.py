import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import seaborn as sns



%matplotlib inline
tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')

tips.head()
tips.total_bill.mean()
from numpy import median

sns.barplot(y='total_bill' ,  x='day' , data = tips ,

            hue = 'smoker' , palette = 'coolwarm' , 

            order =['Sat', 'Sun' , 'Thur' , 'Fri' ], 

            estimator = median , ci = 42 , capsize=0.1 , 

           )   # palette = (winter_r , spring), color = green  , saturation=0.4
num = np.random.randn(640)

sns.distplot(num , color = 'red')
label_dist  = pd.Series(num , name = 'variable')

sns.distplot(label_dist,kde= True, hist = True ) 

# vertical = True   , vertical = True , color = 'red',  rug = True (occurence in bottom)
sns.boxplot(x='total_bill' , y='day'  , data = tips)

# sns.boxplot(tips['total_bill']) # single boxplot

sns.swarmplot(x='total_bill' , y='day'  , data = tips)
tips[['total_bill']].describe()
import pandas as pd

import seaborn as sns

iris = pd.read_csv('../input/iris/Iris.csv')

sns.boxplot(data = iris , orient='horizontal')
sns.stripplot(x=tips['total_bill'] , color= 'red') # one varriables
sns.stripplot(x='day' , y='total_bill', hue = 'sex',\

              data= tips , jitter =0.3 , linewidth=0 ,\

              dodge = True , marker = '^' , size=7) 

# one varriables , jitter = distribution of points , linwidth with of contour point , dodge separte hue var

# edgecolor='red' , (alpha=0.5 transparence)
x = sns.PairGrid(iris)

x = x.map(plt.scatter)
x = sns.PairGrid(iris , hue='Species')

x = x.map_diag(plt.hist ) # , histtype='step' (transparence) , linewidth=3 

x = x.map_offdiag(plt.scatter)

x = x.add_legend()
iris.corr()

x = sns.PairGrid(iris, vars=['PetalLengthCm','PetalWidthCm'], hue='Species') # , hue_kws={'marker':['^','D','-']}

x = x.map(plt.scatter) 
x = sns.PairGrid(iris, x_vars=['PetalWidthCm','PetalLengthCm'] ,

                       y_vars=['PetalLengthCm','SepalLengthCm'] ,

                        hue='Species')

x = x.map(plt.scatter)
x = sns.PairGrid(iris , hue='Species')

x = x.map_diag(plt.hist ) # , histtype='step' (transparence) , linewidth=3 

x = x.map_upper(plt.scatter)

x = x.map_lower(sns.kdeplot)
sns.violinplot(x='day' , y='total_bill', data= tips , hue='smoker' ,

               split=True , inner='quartile') 

#sns.violinplot(tips['total_bill'])  # one variable, scal='count' # inner stick, bw=0.4
flights = pd.read_csv("../input/flights/flights.csv")

flights.head()
flights = flights.pivot('month','year', 'passengers')

sns.clustermap(flights , cmap='coolwarm' ,

               col_cluster=False , row_cluster=False,

              linewidth=1 , figsize=(8,8)) # , standard_scale=0  ,z_score=0 
sns.heatmap(flights , annot=True  , fmt='d' , center=flights.loc['July',1960])



# ,vmin=0 , vmax=1  , cbar=False
tips.head()
x = sns.FacetGrid(tips, row='smoker',col='time')

x= x.map(plt.hist , 'total_bill' )



#color = green , bins=10 = epeceur de carreau
x = sns.FacetGrid(tips,col='time', hue='smoker')

x= x.map(plt.scatter , 'total_bill','tip' )
x = sns.FacetGrid(tips, row='smoker', col='time')

x= x.map(sns.regplot , 'total_bill','tip' )



#sns.boxplot  , (,aspect=3 , size=6, longuer de plot)

# ...'tip').add_legend()
ti = tips[['total_bill', 'tip']].corr()

ti.corr()
versicolor = iris.loc[iris.Species=='Iris-versicolor']

setosa = iris.loc[iris.Species=='Iris-setosa']

sns.kdeplot(setosa.PetalLengthCm , setosa.PetalWidthCm)

sns.kdeplot(versicolor.PetalLengthCm , versicolor.PetalWidthCm)



#, shade=True, Label='setosa', cmap="Blues", shade_lowest=False
iris.head()
sns.jointplot(x='SepalLengthCm' , y = 'SepalWidthCm' , data =iris ,kind='kde' )

# kind = hex , reg , (ratio = 3 , size=4)
sns.regplot(x='SepalLengthCm' , y = 'SepalWidthCm' , data =iris)
sns.pairplot(iris , hue='Species')

# ,vars=['sepal_length','sepal_width']

# , x_vars=['sepal_length','sepal_width'] , y_vars=['sepal_length','sepal_width'] 

# kind = hex , reg , kde