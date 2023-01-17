#this is e2e  visualzation for Data science  so thag this can be used  as  a reference  for other  places  %
%matplotlib inline 

import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

##Utilizing all the cores of  the cpu 

from multiprocessing import Pool









def plotfig():

    ## creating figures and axes   lets give them some var names for future use 

    ax= plt.axes()

    fig=plt.figure()

    # lets fix the type of plot 

    plt.style.use('seaborn-whitegrid')

    



ax= plt.axes()    

points = np.linspace(0,10,1000)



ax.plot(points,np.sin(points))

#other alt is plt.plot 



# plt.xlabel() → ax.set_xlabel()

# plt.ylabel() → ax.set_ylabel()

# plt.xlim() → ax.set_xlim()

# plt.ylim() → ax.set_ylim()

# plt.title() → ax.set_title()



# so now using all the features of plot 

#other alt is plt.plot rather that useing the as shown above 

points = np.linspace(0,10,1000)

plt.plot(points,np.sin(points),'-g')

def sine_plot():

    plt.title('sine curve',color='red')

    plt.xlabel('x',color='red')

    plt.ylabel('sine(x)',color='red')

    plt.axis('equal');

    

sine_plot()

#now the important  chart type is  scattter chart as this can be used in the clustering ot scattered datas points and thus is quiet important

# lets  scatter the point so that points are seperated 

points = np.linspace(0,10,10)

a=plt.plot(points, np.sin(points),'o',color='red')

print("too many points scattered  ",a)

##call the  fun def above to plot here graph wiith styles 

sine_plot()



#lets plot some more complicated scatter chart with multiple properties

plt.plot(points, np.sin(points),'-ok',color='red')



sine_plot()
## one more:

plt.plot(points , '-p', color='gray',

         markersize=15, linewidth=2,

         markerfacecolor='white',

         markeredgecolor='gray',

         markeredgewidth=2)
## using PLT.SCATTER TO PLOT 



#plt.plot should be preferred over plt.scatter for large datasets



## NOTE THAT :The primary difference of plt.scatter from plt.plot is that it can be used to create scatter plots where the properties of each individual point (size, face color, edge color, etc.) can be individually controlled or mapped to data
rndstate = np.random.RandomState(0)

x = rndstate.randn(10)

y = rndstate.randn(10)

colors = rndstate.rand(10)

sizes = 2000 * rndstate.rand(10)



plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,

            cmap='viridis')

plt.colorbar();  # show color scale
##now lets see errorbars visualization :

x = np.linspace(0, 10, 50)

dy = 0.5

y = np.sin(x) + dy * np.random.randn(50)



plt.errorbar(x, y, yerr=dy, fmt='o', color='red',

             ecolor='pink', elinewidth=3, capsize=1);
##for contineous regression we can use :Gaussian process regression:::::::::::::::keeping it out of scope here 
#Visualizing a Three-Dimensional Function: Density and Contour Plots

##Histograms, Binnings, and Density

hist_data =np.random.rand(100)

plt.hist(hist_data)

##lets addd some pmore properties 

plt.hist(hist_data, bins=30, normed=True, alpha=0.5,

         histtype='stepfilled', color='steelblue',

         edgecolor='none');
##using stefill:Use this to compare the histograms of different comparisions 

d1 = np.random.normal(0, 0.8, 1000)

d2 = np.random.normal(-2, 1, 1000)

d3 = np.random.normal(3, 2, 1000)



kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)



plt.hist(d1, **kwargs)

plt.hist(d2, **kwargs)

plt.hist(d3, **kwargs);
#plotitng 2 d histograms 

## so here we use 2d=plt.hist2d instead of 1d= plt.hist 

#### Also: evaluating densities in multiple dimensions is kernel density estimation (KDE) we can do it later once we need it 

##using multiple subplots : use it to show muliple relations brtween the same set od data but having different featuress 

fig=plt.figure()

## lets plot the multiple charts

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],

                   xticklabels=[], ylim=(-1.2, 1.2))

ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],

                   ylim=(-1.2, 1.2))

## now lets create a sine wave 

x=np.linspace(1,10,100)

ax1.plot(np.sin(x))

ax2.plot(np.sin(x))
##to make the chats more informative we can use tests and aaanotation  in the charts:

fig, ax = plt.subplots()



x = np.linspace(0, 20, 1000)

ax.plot(x, np.cos(x))

ax.axis('equal')



ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),

            arrowprops=dict(facecolor='black', shrink=0.05))



ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="angle3,angleA=0,angleB=-90"));
## things pending: 3d plotting , plotting gre data , plot using seaborn in depth 
