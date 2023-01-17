import pandas as pd

import numpy as np

%matplotlib inline
from plotly import __version__

print(__version__)
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected = True)
cf.go_offline()
#Getting Data

df = pd.DataFrame(np.random.randn(100,4),columns = 'A B C D'.split())
df.head()
df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2
df.plot()
df.iplot()
df.iplot(kind = 'scatter',x = 'A',y = 'B',mode = 'markers',color = 'black')
df2.iplot(kind = 'bar',x = 'Category',y = 'Values')
df.iplot(kind = 'bar')
df.sum().iplot(kind = 'bar')
df.iplot(kind = 'box')
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3
df3.iplot(kind = 'surface',colorscale='rdylbu')
df['A'].iplot(kind = 'hist',bins = 50)
df.iplot(kind = 'hist')
df[['A','B']].iplot(kind = 'spread')
df.iplot(kind = 'bubble',x = 'A',y = 'B',size = 'C')

#used for finding gdp,and in UN data plots
df.scatter_matrix() #all columns need to be numeric to make it work