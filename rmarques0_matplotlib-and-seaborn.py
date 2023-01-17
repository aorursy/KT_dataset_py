import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

x = np.linspace(0, 5, 11)

y = x ** 2
x
y
plt.plot(x, y, 'r') # 'r' is the color red

plt.xlabel('X Axis Title Here')

plt.ylabel('Y Axis Title Here')

plt.title('String Title Here')

plt.show()
# plt.subplot(nrows, ncols, plot_number)

plt.subplot(1,2,1)

plt.plot(x, y, 'r--') # More on color options later

plt.subplot(1,2,2)

plt.plot(y, x, 'g*-');
# Create Figure (empty canvas)

fig = plt.figure()



# Add set of axes to figure

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)



# Plot on that set of axes

axes.plot(x, y, 'b')

axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods

axes.set_ylabel('Set y Label')

axes.set_title('Set Title');
# Creates blank canvas

fig = plt.figure()



axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes



# Larger Figure Axes 1

axes1.plot(x, y, 'b')

axes1.set_xlabel('X_label_axes2')

axes1.set_ylabel('Y_label_axes2')

axes1.set_title('Axes 2 Title')



# Insert Figure Axes 2

axes2.plot(y, x, 'r')

axes2.set_xlabel('X_label_axes2')

axes2.set_ylabel('Y_label_axes2')

axes2.set_title('Axes 2 Title');
# Use similar to plt.figure() except use tuple unpacking to grab fig and axes

fig, axes = plt.subplots()



# Now use the axes object to add stuff to plot

axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
# Empty canvas of 1 by 2 subplots

fig, axes = plt.subplots(nrows=1, ncols=2)
# Axes is an array of axes to plot on

axes
for ax in axes:

    ax.plot(x, y, 'b')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



# Display the figure object    

fig
fig, axes = plt.subplots(nrows=1, ncols=2)



for ax in axes:

    ax.plot(x, y, 'g')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.set_title('title')



fig    

plt.tight_layout()
fig = plt.figure(figsize=(8,4), dpi=100)
fig, axes = plt.subplots(figsize=(12,3))



axes.plot(x, y, 'r')

axes.set_xlabel('x')

axes.set_ylabel('y')

axes.set_title('title');
fig.savefig("filename.png")
fig.savefig("filename.png", dpi=200)
ax.set_title("title");
ax.set_xlabel("x")

ax.set_ylabel("y");
fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, label="x**2")

ax.plot(x, x**3, label="x**3")

ax.legend();
# Lots of options....



ax.legend(loc=1) # upper right corner

ax.legend(loc=2) # upper left corner

ax.legend(loc=3) # lower left corner

ax.legend(loc=4) # lower right corner



# .. many more options are available



# Most common to choose

ax.legend(loc=0) # let matplotlib decide the optimal location

fig
# MATLAB style line color and style 

fig, ax = plt.subplots()

ax.plot(x, x**2, 'b.-') # blue line with dots

ax.plot(x, x**3, 'g--'); # green dashed line
fig, ax = plt.subplots()



ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant

ax.plot(x, x+2, color="#8B008B")        # RGB hex code

ax.plot(x, x+3, color="#FF8C00");      # RGB hex code
fig, ax = plt.subplots(figsize=(12,6))



ax.plot(x, x+1, color="red", linewidth=0.25)

ax.plot(x, x+2, color="red", linewidth=0.50)

ax.plot(x, x+3, color="red", linewidth=1.00)

ax.plot(x, x+4, color="red", linewidth=2.00)



# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’

ax.plot(x, x+5, color="green", lw=3, linestyle='-')

ax.plot(x, x+6, color="green", lw=3, ls='-.')

ax.plot(x, x+7, color="green", lw=3, ls=':')



# custom dash

line, = ax.plot(x, x+8, color="black", lw=1.50)

line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...



# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...

ax.plot(x, x+ 9, color="blue", lw=3, ls='-', marker='+')

ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')

ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')

ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')



# marker size and color

ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)

ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)

ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")

ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 

        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green");
fig, axes = plt.subplots(1, 3, figsize=(12, 4))



axes[0].plot(x, x**2, x, x**3)

axes[0].set_title("default axes ranges")



axes[1].plot(x, x**2, x, x**3)

axes[1].axis('tight')

axes[1].set_title("tight axes")



axes[2].plot(x, x**2, x, x**3)

axes[2].set_ylim([0, 60])

axes[2].set_xlim([2, 5])

axes[2].set_title("custom axes range");
plt.scatter(x,y);
from random import sample

data = sample(range(1, 1000), 100)

plt.hist(data)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]



# rectangular box plot

plt.boxplot(data,vert=True,patch_artist=True);
fig, axes = plt.subplots(1, 2, figsize=(10,4))

      

axes[0].plot(x, x**2, x, np.exp(x))

axes[0].set_title("Normal scale")



axes[1].plot(x, x**2, x, np.exp(x))

axes[1].set_yscale("log")

axes[1].set_title("Logarithmic scale (y)");
fig, ax = plt.subplots(figsize=(10, 4))



ax.plot(x, x**2, x, x**3, lw=2)



ax.set_xticks([1, 2, 3, 4, 5])

ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)



yticks = [0, 50, 100, 150]

ax.set_yticks(yticks)

ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels
fig, ax = plt.subplots(1, 1)

      

ax.plot(x, x**2, x, np.exp(x))

ax.set_title("scientific notation")



ax.set_yticks([0, 50, 100, 150])



from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)

formatter.set_scientific(True) 

formatter.set_powerlimits((-1,1)) 

ax.yaxis.set_major_formatter(formatter)
# distance between x and y axis and the numbers on the axes

matplotlib.rcParams['xtick.major.pad'] = 5

matplotlib.rcParams['ytick.major.pad'] = 5



fig, ax = plt.subplots(1, 1)

      

ax.plot(x, x**2, x, np.exp(x))

ax.set_yticks([0, 50, 100, 150])



ax.set_title("label and axis spacing")



# padding between axis label and axis numbers

ax.xaxis.labelpad = 5

ax.yaxis.labelpad = 5



ax.set_xlabel("x")

ax.set_ylabel("y");
# restore defaults

matplotlib.rcParams['xtick.major.pad'] = 3

matplotlib.rcParams['ytick.major.pad'] = 3
fig, ax = plt.subplots(1, 1)

      

ax.plot(x, x**2, x, np.exp(x))

ax.set_yticks([0, 50, 100, 150])



ax.set_title("title")

ax.set_xlabel("x")

ax.set_ylabel("y")



fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9);
fig, axes = plt.subplots(1, 2, figsize=(10,3))



# default grid appearance

axes[0].plot(x, x**2, x, x**3, lw=2)

axes[0].grid(True)



# custom grid appearance

axes[1].plot(x, x**2, x, x**3, lw=2)

axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
fig, ax = plt.subplots(figsize=(6,2))



ax.spines['bottom'].set_color('blue')

ax.spines['top'].set_color('blue')



ax.spines['left'].set_color('red')

ax.spines['left'].set_linewidth(2)



# turn off axis spine to the right

ax.spines['right'].set_color("none")

ax.yaxis.tick_left() # only ticks on the left side
fig, ax1 = plt.subplots()



ax1.plot(x, x**2, lw=2, color="blue")

ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")

for label in ax1.get_yticklabels():

    label.set_color("blue")

    

ax2 = ax1.twinx()

ax2.plot(x, x**3, lw=2, color="red")

ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")

for label in ax2.get_yticklabels():

    label.set_color("red")
fig, ax = plt.subplots()



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0



ax.yaxis.set_ticks_position('left')

ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0



xx = np.linspace(-0.75, 1., 100)

ax.plot(xx, xx**3);
n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1, 4, figsize=(12,3))



axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))

axes[0].set_title("scatter")



axes[1].step(n, n**2, lw=2)

axes[1].set_title("step")



axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)

axes[2].set_title("bar")



axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);

axes[3].set_title("fill_between");
fig, ax = plt.subplots()



ax.plot(xx, xx**2, xx, xx**3)



ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")

ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green");
fig, ax = plt.subplots(2, 3)

fig.tight_layout()
fig = plt.figure()

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)

ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)

ax4 = plt.subplot2grid((3,3), (2,0))

ax5 = plt.subplot2grid((3,3), (2,1))

fig.tight_layout()
import matplotlib.gridspec as gridspec
fig = plt.figure()



gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1,2,1])

for g in gs:

    ax = fig.add_subplot(g)

    

fig.tight_layout()
fig, ax = plt.subplots()



ax.plot(xx, xx**2, xx, xx**3)

fig.tight_layout()



# inset

inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height



inset_ax.plot(xx, xx**2, xx, xx**3)

inset_ax.set_title('zoom near origin')



# set axis range

inset_ax.set_xlim(-.2, .2)

inset_ax.set_ylim(-.005, .01)



# set axis tick locations

inset_ax.set_yticks([0, 0.005, 0.01])

inset_ax.set_xticks([-0.1,0,.1]);
alpha = 0.7

phi_ext = 2 * np.pi * 0.5



def flux_qubit_potential(phi_m, phi_p):

    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)
phi_m = np.linspace(0, 2*np.pi, 100)

phi_p = np.linspace(0, 2*np.pi, 100)

X,Y = np.meshgrid(phi_p, phi_m)

Z = flux_qubit_potential(X, Y).T
fig, ax = plt.subplots()



p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())

cb = fig.colorbar(p, ax=ax)
fig, ax = plt.subplots()



im = ax.imshow(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])

im.set_interpolation('bilinear')



cb = fig.colorbar(im, ax=ax)
fig, ax = plt.subplots()



cnt = ax.contour(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure(figsize=(14,6))



# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot

ax = fig.add_subplot(1, 2, 1, projection='3d')



p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)



# surface_plot with color grading and color bar

ax = fig.add_subplot(1, 2, 2, projection='3d')

p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)

cb = fig.colorbar(p, shrink=0.5)
fig = plt.figure(figsize=(8,6))



ax = fig.add_subplot(1, 1, 1, projection='3d')



p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
fig = plt.figure(figsize=(8,6))



ax = fig.add_subplot(1,1,1, projection='3d')



ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)

cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=matplotlib.cm.coolwarm)

cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=matplotlib.cm.coolwarm)

cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=matplotlib.cm.coolwarm)



ax.set_xlim3d(-np.pi, 2*np.pi);

ax.set_ylim3d(0, 3*np.pi);

ax.set_zlim3d(-np.pi, 2*np.pi);
import seaborn as sns

%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
sns.distplot(tips['total_bill'])

# Safe to ignore warnings
sns.distplot(tips['total_bill'],kde=False,bins=30)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
sns.pairplot(tips)
sns.pairplot(tips,hue='sex',palette='coolwarm')
sns.rugplot(tips['total_bill'])
# Don't worry about understanding this code!

# It's just for the diagram below

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats



#Create dataset

dataset = np.random.randn(25)



# Create another rugplot

sns.rugplot(dataset);



# Set up the x-axis for the plot

x_min = dataset.min() - 2

x_max = dataset.max() + 2



# 100 equally spaced points from x_min to x_max

x_axis = np.linspace(x_min,x_max,100)



# Set up the bandwidth, for info on this:

url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'



bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2





# Create an empty kernel list

kernel_list = []



# Plot each basis function

for data_point in dataset:

    

    # Create a kernel for each point and append to list

    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)

    kernel_list.append(kernel)

    

    #Scale for plotting

    kernel = kernel / kernel.max()

    kernel = kernel * .4

    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)



plt.ylim(0,1)
# To get the kde plot we can sum these basis functions.



# Plot the sum of the basis function

sum_of_kde = np.sum(kernel_list,axis=0)



# Plot figure

fig = plt.plot(x_axis,sum_of_kde,color='indianred')



# Add the initial rugplot

sns.rugplot(dataset,c = 'indianred')



# Get rid of y-tick marks

plt.yticks([])



# Set title

plt.suptitle("Sum of the Basis Functions")
sns.kdeplot(tips['total_bill'])

sns.rugplot(tips['total_bill'])
sns.kdeplot(tips['tip'])

sns.rugplot(tips['tip'])
import seaborn as sns

%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
sns.distplot(tips['total_bill'])

# Safe to ignore warnings
sns.distplot(tips['total_bill'],kde=False,bins=30)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
sns.pairplot(tips)
sns.pairplot(tips,hue='sex',palette='coolwarm')
sns.rugplot(tips['total_bill'])
# Don't worry about understanding this code!

# It's just for the diagram below

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats



#Create dataset

dataset = np.random.randn(25)



# Create another rugplot

sns.rugplot(dataset);



# Set up the x-axis for the plot

x_min = dataset.min() - 2

x_max = dataset.max() + 2



# 100 equally spaced points from x_min to x_max

x_axis = np.linspace(x_min,x_max,100)



# Set up the bandwidth, for info on this:

url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'



bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2





# Create an empty kernel list

kernel_list = []



# Plot each basis function

for data_point in dataset:

    

    # Create a kernel for each point and append to list

    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)

    kernel_list.append(kernel)

    

    #Scale for plotting

    kernel = kernel / kernel.max()

    kernel = kernel * .4

    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)



plt.ylim(0,1)
# To get the kde plot we can sum these basis functions.



# Plot the sum of the basis function

sum_of_kde = np.sum(kernel_list,axis=0)



# Plot figure

fig = plt.plot(x_axis,sum_of_kde,color='indianred')



# Add the initial rugplot

sns.rugplot(dataset,c = 'indianred')



# Get rid of y-tick marks

plt.yticks([])



# Set title

plt.suptitle("Sum of the Basis Functions")
sns.kdeplot(tips['total_bill'])

sns.rugplot(tips['total_bill'])
sns.kdeplot(tips['tip'])

sns.rugplot(tips['tip'])
import seaborn as sns

%matplotlib inline
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')
tips.head()
flights.head()
tips.head()
# Matrix form for correlation data

tips.corr()
sns.heatmap(tips.corr())
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)
flights.pivot_table(values='passengers',index='month',columns='year')
pvflights = flights.pivot_table(values='passengers',index='month',columns='year')

sns.heatmap(pvflights)
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)
sns.clustermap(pvflights)
# More options to get the information a little clearer like normalization

sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
iris = sns.load_dataset('iris')
iris.head()
# Just the Grid

sns.PairGrid(iris)
# Then you map to the grid

g = sns.PairGrid(iris)

g.map(plt.scatter)
# Map to upper,lower, and diagonal

g = sns.PairGrid(iris)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)
sns.pairplot(iris)
sns.pairplot(iris,hue='species',palette='rainbow')
tips = sns.load_dataset('tips')
tips.head()
# Just the Grid

g = sns.FacetGrid(tips, col="time", row="smoker")
g = sns.FacetGrid(tips, col="time",  row="smoker")

g = g.map(plt.hist, "total_bill")
g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')

# Notice hwo the arguments come after plt.scatter call

g = g.map(plt.scatter, "total_bill", "tip").add_legend()
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = sns.JointGrid(x="total_bill", y="tip", data=tips)

g = g.plot(sns.regplot, sns.distplot)
import seaborn as sns

%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
sns.lmplot(x='total_bill',y='tip',data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm')
# http://matplotlib.org/api/markers_api.html

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm',

           markers=['o','v'],scatter_kws={'s':100})
sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')
sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm')
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm',

          aspect=0.6,size=8)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

tips = sns.load_dataset('tips')
sns.countplot(x='sex',data=tips)
sns.set_style('white')

sns.countplot(x='sex',data=tips)
sns.set_style('ticks')

sns.countplot(x='sex',data=tips,palette='deep')
sns.countplot(x='sex',data=tips)

sns.despine()
sns.countplot(x='sex',data=tips)

sns.despine(left=True)
# Non Grid Plot

plt.figure(figsize=(12,3))

sns.countplot(x='sex',data=tips)
# Grid Type Plot

sns.lmplot(x='total_bill',y='tip',size=2,aspect=4,data=tips)
sns.set_context('poster',font_scale=4)

sns.countplot(x='sex',data=tips,palette='coolwarm')
sns.puppyplot()