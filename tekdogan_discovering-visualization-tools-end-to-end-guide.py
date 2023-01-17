import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Configure the graph.
ts = pd.Series(np.random.randn(300), index=pd.date_range('1/1/2017', periods=300))

#Return cumulative sum over requested axis.
ts = ts.cumsum()

#Plot the graph.
ts.plot()
#Calling np.random.randn four times to plot four different curves in same graph.
#Using index=ts.index (past index configurations)
df = pd.DataFrame(np.random.randn(300, 4), index=ts.index, columns=list('ABCD'))

df = df.cumsum()

plt.figure()

df.plot()
plt.figure()

df.plot(legend = False)
plt.figure()

df.plot(subplots = True, figsize = (8,8))
plt.figure()

ts.plot(style = 'k--')
plt.figure()

ts.plot(x_compat = True)
plt.figure()

df.iloc[2].plot(kind='bar');
#The code line below configures four variables (a,b,c,d) through 10-length axis.
df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

#If we just call df2.plot(), it will be plotting a line graph.
df2.plot()
#But if we call df2.plot.bar() instead, it plots the graph in bars.
df2.plot.bar()
df2.plot.bar(stacked=True)
df2.plot.barh()
df2.plot.barh(stacked=True)
#Configure 3 variables v1,v2 and v3
dfHorizontal = pd.DataFrame({'v1' : np.random.randn(500) - 1 , 'v2' : np.random.randn(500) , 'v3' : np.random.randn(500) + 1}, columns = ['v1','v2','v3'])

plt.figure()

#Transparency helps us to eliminate the overlap problem
dfHorizontal.plot.hist(alpha=0.5)
plt.figure()

dfHorizontal.plot.hist(stacked=True, bins=15)
plt.figure()

dfHorizontal.diff().hist(bins=50)
dfBoxPlot = pd.DataFrame(np.random.rand(12,3), columns = ['var1','var2','var3'])

dfBoxPlot.plot.box()
c = dict(boxes = 'red', whiskers = 'aqua', medians = 'green', caps = 'orange')

dfBoxPlot.plot.box(color = c)
dfAreaPlot = pd.DataFrame(np.random.rand(10,5), columns = ['var1','var2','var3','var4','var5'])

dfAreaPlot.plot.area()
dfAreaPlot.plot.area(stacked = False, alpha = 0.3)
dfScatter = pd.DataFrame(np.random.rand(100,2), columns = ['var1','var2'])

#Parameters x and y have to be denoted while plotting.
dfScatter.plot.scatter(x = 'var1', y = 'var2', color = 'violet')
#Configuring a thousand random points.
dfHexa = pd.DataFrame(np.random.rand(1000,2), columns = ['var1', 'var2'])

#Plotting them with hexbin.
dfHexa.plot.hexbin(x = 'var1', y = 'var2', gridsize = 20)
pieSeries = pd.Series(np.random.rand(4), index=['var1', 'var2', 'var3', 'var4'], name='series')

pieSeries.plot.pie(figsize=(6, 6))
pieDataFrame = pd.DataFrame(np.random.rand(4, 2), index=['var1', 'var2', 'var3', 'var4'], columns=['col1', 'col2'])

pieDataFrame.plot.pie(subplots=True, figsize=(12, 6))
pieDataFrame.plot.pie(subplots = True, labels = ['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4'], colors = ['blue','gray','aqua','magenta'], autopct = '%.2f', figsize=(12, 6))
#Passing 0.15 per variable, 0.75 as total
pieSeries = pd.Series([0.15] * 5, index=['var1', 'var2', 'var3', 'var4', 'var5'], name='series')

pieSeries.plot.pie(figsize=(6, 6))
#Import scatter_matrix
from pandas.plotting import scatter_matrix

#Import dataset
data = pd.read_csv('../input/Iris.csv')

scatter_matrix(data, alpha = 0.8, figsize = (10,10))
from pandas.plotting import andrews_curves

plt.figure()

andrews_curves(data, 'Species')
from pandas.plotting import parallel_coordinates

plt.figure()

parallel_coordinates(data, 'Species')
