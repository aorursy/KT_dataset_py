import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
ts1=pd.Series([x*x*x*x for x in range(1,30)])

ts1.plot()
ts2=pd.Series([x*x*x for x in range(1,30)])

ts2.columns=['no qube']

ts2.plot(title='line plot')

ts3=pd.Series([x*x for x in range(1,30)])

ts3.plot()
df=pd.DataFrame()

df.insert(0, "ts1", ts1)

df.insert(1, "ts2", ts2)

df.insert(2, "ts3", ts3)

df.plot()
plt.figure();

df.plot(color=['red','green','blue'] );



df = pd.DataFrame(data=np.random.randn(1000, 4), columns=list('ABCD'))

df = df.cumsum()

df.plot()

df.plot(subplots=True)
#bar chart
df.iloc[5].plot(kind='bar')# single Row plot
df.plot(kind='bar')# whole dataframe plot
#2 multiple bar chart

df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df2.plot.bar();
#stacked bar chart

df2.plot.bar(stacked=True);

#stacked horizontal bar chart

df2.plot.barh(stacked=True);
#hishtogram ploting

df4 = pd.DataFrame({'a': np.random.randn(1000) , 'b': np.random.randn(1000),'c': np.random.randn(1000)}, columns=['a', 'b', 'c'])

df4['a'].hist()

df4['a'].hist(grid=False)

#plot histogram of whole dataframe with subplot

df4.hist()
#plot all column in one histogram

df4 = pd.DataFrame({'a': np.random.randn(1000) , 'b': np.random.randn(1000),'c': np.random.randn(1000)}, columns=['a', 'b', 'c'])

plt.figure();

df4['a'].hist()

df4['b'].hist(color='blue')

df4['c'].hist(color='red')

#add opecity for more clear picture

df4.plot.hist(alpha=0.5,grid=True)

df4.plot.hist( alpha=0.3)

#horizontal Ploting

df4['a'].plot.hist(orientation='horizontal')
#box plots

df = pd.DataFrame(np.random.rand(10), columns=['A'])

df['A'].plot.box()
#plot multiple column 

df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

df.plot.box()
#multiple column plot with difrent position

df.plot.box( positions=[1, 4, 5, 6, 8])

#change Color

color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')

df.plot.box(color=color)
#vertical box plot

df.plot.box(vert=False)

df.plot.box(vert=False, positions=[1, 4, 5, 6, 8])
#Enable Gride

df = pd.DataFrame(np.random.rand(10,5))

plt.figure();

df.boxplot(grid=True)
#grouping of column

df = pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )

df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])

plt.figure();

bp = df.boxplot(by='X')
#area chart

df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df.plot.area();

#change Color

df.plot.area(color=['green','red']);
#scatter Plot

df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c','d'])

df.plot.scatter(x='a', y='b');
#multiple axis scatter plot

ax = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1');

df.plot.scatter(x='c', y='d', color='Green', label='Group 2', ax=ax);
#Pie chart

series = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')

series.plot.pie(figsize=(6, 6))
#compair two pi chart

df = pd.DataFrame(np.random.rand(4, 2), index=['a', 'b', 'c', 'd'], columns=['x', 'y'])

df.plot.pie(subplots=True, figsize=(8, 4))



#pi chart with lesstan 100%

series = pd.Series([0.1] * 4, index=['a', 'b', 'c', 'd'], name='series2')

series.plot.pie(figsize=(6, 6))
