# Import matplotlib

import matplotlib.pyplot as plt

# Import toolkit for 3D plots (in matplotlib)

from mpl_toolkits.mplot3d import Axes3D

# This is just to avoid warning related to conversion of DateTime conversion when plotting

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# To format ticks

import matplotlib.dates as mdates

# This magic will ensure that the plots are in Jupyter in high quality=vector drawing

# see https://stackoverflow.com/questions/25412513/inline-images-have-low-quality

%config InlineBackend.figure_format = "svg"
# Set matplotlib default font size (inside this notebook)

# See https://matplotlib.org/users/customizing.html

plt.rcParams.update({'font.size': 9})
import numpy as np

import pandas as pd

import scipy as sp

# To scale data before plotting

from sklearn.preprocessing import StandardScaler

# Load scikit-learn data set

from sklearn.datasets import load_boston
boston = load_boston();

df = pd.DataFrame(boston.data,columns=boston.feature_names)

df = df.assign(target=boston.target);

df.head(3)
# figure with `fig` ID (=object) and size of 4x2.5 inches (width x height)

fig = plt.figure(figsize=(4,2.5));

# Axes inside this figure. Set to full size (from zero to 100% of the figure)

ax = fig.add_axes([0,0,1,1]);

# Plot a line inside this axes 

# it creates "2D object" even though one might think of it 1D where the y depends only on one x

l = ax.plot([1,1,2,3,3],[-1,1,0,1,-1]);

# can change/set some properties of `ax`

ax.set_title("matplotlib: Figure & Axes example");

ax.set_xticks([]);ax.set_yticks([]); # just to turn off x/y ticks (shown coordinates)

# Do not show x/y axes frame

ax.axis("off");



# plt.show() # would need to call outside Jupyter
# set figure size to 6 by 3.5 inches

plt.figure(figsize=(6,3.5))



# scatter plot with black ("k") dots (".") inside the "current" axes

plt.plot(df.target,df.RM,"k.");

# same as

#plt.scatter(df.target,df.RM,c="k",marker=".")



# fit line = degree one polynomial

p = np.polyfit(df.target,df.RM,deg=1)



# Plot inside the same figure (in the same notebook cell)

plt.plot(df.target,np.polyval(p,df.target),"r-",linewidth=2);



# Add description

plt.xlabel("price",fontsize=10);

plt.ylabel("number of rooms",fontsize=10)

plt.title("matplotlib scatter and fit plot");

plt.legend(["input","fit"]);
# Generate time vector (same length as x1) from first to forth month of 2010 with `D`aily sampling

t = np.arange("2010-01-01","2010-04-01", dtype="datetime64[D]")

# Generate `y` value (use `t.astype('float')` to convert to float days)

y = np.sin(2*np.pi*1/30*t.astype("float")+10);

# add some noise

y1 = y + np.random.randn(y.size)/10;
plt.figure(figsize=(8,4))

ax1 = plt.subplot(2,1,1);

ax1.plot(t,y,"k-");

ax2 = plt.subplot(2,1,2,sharex=ax1);

ax2.plot(t,y1,"bo");



# PyPlot plots time ticks automatically sometimes very strangly 

# Set to monthly step 

ax1.xaxis.set_major_locator(mdates.MonthLocator())

# can also use:

# ax1.set_xticks(np.arange("2010-01-01","2010-04-01",dtype="datetime64[M]"));



# Format to "YYYY-MM-DD" 

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d")) # every month



# Set xlimit. Add one day to the last value of time vector

ax1.set_xlim((t[0],t[-1]+np.timedelta64(1,"D")))



# As ax1 and ax2 are synced, need to set only ax1. 

# For the same reason `ax1.set_xticklabels([])` 

# would not work if we want to have labels only in the second plot 

# Turn of the xtick label visibility for `ax1`. 

for xtick in ax1.get_xticklabels():

    xtick.set_visible(False)

    

# Set legend

ax1.legend(["target/label"],loc=4,facecolor=None);

ax2.legend(["feature"],loc=4,frameon=False);

ax1.set_title("matplotlib time series subplot with shared time axis");
fig = plt.figure(figsize=(10,5));

# need to create 3D axes

ax = fig.add_subplot(111, projection="3d");



# plot 3D using features as coordinates and target as color. 

# This particular dependence/arrangement does not make much sense it just shows how it works.

ax.scatter(df.RM,df.LSTAT,df.DIS, # x,y,z

           c=df.target, # color

           marker="o",s=40); # marker type and size (can be also variable)



# Set z coordinate limit manually

ax.set_zlim([1,10]);



# Show labels

ax.set_xlabel("x = RM");

ax.set_ylabel("y = MEDV");

ax.set_zlabel("z = DIS");
def plotcolumns(datain,pltfce,maxcol=4,xvec=np.array([]),colname=np.array([])):

    """

    Plot `n` dimensional data (`datain`) inside one figure segmented into a given number 

    of columns (`maxcol`) and derived number of rows

    

    **Input**

    * `datain`: input np.array/pd.DataFrame with `n` columns to be plotted

    * `pltfce`: pyplot plotting function, e.g., `hist`, `scatter`, `plot`

    * `maxcol`: maximum number of subplot columns (4 by default)

    * `xvec`: optional x-axis values. If used, must be same size as number of rows in `datain`

    * `colname`: column names (optional). See `getcolname` function

    

    **Example**

    ```

    plotdata = np.array([[1,10,100,1000,10000],[2,20,200,2000,20000],[3,30,300,3000,30000]]);

    plt.figure(figsize=(8,4));

    plotcolumns(plotdata,plt.plot,maxcol=3)

    

    # Or use dataframe (column names will be used automatically)

    plotdf = pd.DataFrame(plotdata,columns=["one","two","three","four","five"]);

    plotcolumns(plotdf,plt.scatter,xvec=plotdf.index);

    ```

    """

    # Derive number of columns and rows. Use the maxcol setting as threshold

    n = datain.shape[1];

    ncol = n if n <= maxcol else maxcol;

    nrow = np.int(np.ceil(n/ncol));

    

    # Get column names to be used as legend. 

    # See definition in following cell = set "column_X" if array or `colname` not defined

    colname = getcolname(datain,colname);

    

    for i in range(0,n):

        # Create new subplot for each column

        plt.subplot(nrow,ncol,i+1);

        

        # plot with or without x-vector

        if xvec.shape[0] == datain.shape[0]:

            pltfce(xvec,datain.iloc[:,i] if isinstance(datain, pd.DataFrame) else datain[:,i]);

        else:

            pltfce(datain.iloc[:,i] if isinstance(datain, pd.DataFrame) else datain[:,i]);

    

        # Add legend in the current subplot

        plt.legend([colname[i]]);
def getcolname(datain,colname):

    """

    Aux. function to extract column names suitable for `plotcolumns` function

    """

    if isinstance(colname,list): # just to use `size` property

        colname = np.array(colname);

    elif isinstance(datain, pd.DataFrame):

        colname = datain.columns;

        

    if colname.size == datain.shape[1]:

        return colname

    else: # create column name

        return np.array(["column_{}".format(i) for i in range(0,datain.shape[1])])
plt.figure(figsize=(9.5,7))

# Use `lambda` function to set optional number of bins (set to sqrt(x.shape[0]))

plotcolumns(df,lambda x: plt.hist(x,bins="sqrt"));
bplot = StandardScaler().fit_transform(df.values);
plt.figure(figsize=(9,4))

plt.boxplot(bplot,

            meanline=True,showmeans=True); # after scaling, the mean line will be always at zero

plt.gca().set_xticklabels(df.columns);

plt.gca().set_ylabel("scaled values",fontsize=10);

plt.title("matplotlib boxplot");
# Use pandas methods to compute correlation matrix for all columns, 

# round the result to 3 decimals and show the dataframe with background color

df.corr().round(3).style.background_gradient(cmap="viridis")
def corrplot(data,colname=np.array([])):

    """

    Plot correlation matrix 

    

    **Input**

    * `data`: input np.array/pd.DataFrame with `n` columns to be plotted

    * `colname`: column names (optional). See `getcolname` function

    

    **Output**

    * created axes

    

    **Example**

    ```

    ax = corrplot(np.array([[1,2],[2,3],[3,2],[3,3],[4,2]]),colname=["1","2"])

    ```

    """

    # Get column names to be used as x/y tick-labels. 

    colname = getcolname(data,colname);

    

    # Compute correlation matrix. Need to transpose the data to work with numpy.corrcoef

    cormat = np.corrcoef(data.T if not isinstance(data, pd.DataFrame) else data[colname].values.T)

    

    # Create image

    im = plt.imshow(np.corrcoef(data.T));

    

    # Use current axes to add names as tick-labels

    ax = plt.gca();

    for a,b in zip([ax.set_xticks,ax.set_yticks],[ax.set_xticklabels,ax.set_yticklabels]):

        a(np.arange(data.shape[1]));

        b(colname);

    

    # Add colorbar

    ax.figure.colorbar(im, ax=ax)

    

    # return axes

    return ax
# Open new figure

# the y size just controls the colorbar as plot ratio will be in this case (=imshow) always 1:1

plt.figure(figsize=(9,7)) 

# Call the defined function using original column names as ticks

corrplot(df.values,colname=df.columns);

# Add title

plt.title("matplotlib correlation matrix");

# Print specifying resolution and transparency

# plt.savefig("output_figure.png", dpi=600, transparent=True) 