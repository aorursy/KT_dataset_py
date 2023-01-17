# These three statements each loads a package containing new functions for us to use
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# These statements are optional, but will set some of the plotting parameters to look nice
sns.set_style("ticks")
sns.set_context("talk")
halon = pd.read_fwf('../input/demodata/H-1301.txt')
halon.head() # Shows first few rows of a dataframe
halon.info()
halon['decdate'] # Can get values in any column using brackets and column name
# For reference, this would be how to make a date column

date=pd.to_datetime(halon['yyyymmdd'],format="%Y%m%d")
halon = pd.concat([halon,date.to_frame(name="date")],1)
date
halon = halon.sort_values(by='decdate')
halon.head()
plt.figure(figsize=(15,10))    # Ask MatPlotLib to make a figure that is 15 units wide and 10 tall

# Make our plot with the lineplot function
# data, x, and y tell the function what to plot
# hue tells it that these columns are separate series to plot
# lw tells it to make the lines thin
sns_plot = sns.lineplot(data = halon, x='decdate',y='H-1301_C',hue='instr.',lw=.5)


# Alternative format: sns.lineplot(x=halon['decdate'],y=halon['H-1301_C'],hue=halon['instr.'],lw=.5)
plt.figure(figsize=(10,7))    # Ask MatPlotLib to make a figure that is 15 units wide and 10 tall

sns_plot = sns.lineplot(data = halon, x='decdate',y='H-1301_C',hue='instr.',lw=.5) # Same plot as before
sns_plot.set(xlim=(2013,2017),ylim=(3,3.7)) # Set the x and y axis limits
plt.locator_params(axis='x', nbins=4)       # Tell it to plot fewer x ticks 
fig=sns_plot.get_figure()    # New variable that refers to the container for our plot
plt.rc('pdf', fonttype=42)   # This line declares a font that renders as a font in Illustrator (instead of outlines)
fig.savefig("H-1301fig.pdf") # Save as a file. Change format by changing suffix
xl = pd.read_excel('../input/demodata/ImportantExperiment.xlsx')
xl.head()
sns.lineplot(data=xl,x='time (min)',y='absorbance',hue='sample')
sns.despine()    # This takes the top and right borders off
sns.boxplot(data=xl,x='sample',y='absorbance')
sns.stripplot(data=xl,x='sample',y='absorbance', jitter=True)
exp = sns.stripplot(data=xl,x='sample',y='absorbance')
exp = sns.boxplot(data=xl,x='sample',y='absorbance', color="0.9")
plt.figure() # make a new figure container to put data in
sns.violinplot(data=xl,x='sample',y='absorbance')
iris = pd.read_csv("../input/demodata/iris.csv")
iris.head()
iris.info()
sns.boxplot(data=iris, orient='h')
plt.figure(figsize=(7,6))
sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='species')
plt.legend(bbox_to_anchor=(1, .5), loc="upper left") # move the legend
plt.figure(figsize=(15,15)) 
sns.pairplot(data=iris,hue="species")
#  make many plots at once
categories_to_plot = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
for cat in categories_to_plot:
    plt.figure()
    sns.violinplot(data=iris, x="species", y=cat)
# categorical plots
categories_to_plot = ["petal_length", "petal_width", "sepal_length", "sepal_width"]

for cat in categories_to_plot:
    fig, axarr = plt.subplots(1,3, figsize=(25,6))  # Each loop creates a 1x3 grid of plots
                                                  # Each subplot is an entry in the array axarr
        
    sns.violinplot(data=iris, x="species", y=cat, ax=axarr[0]) # ax tells it which subplot to put the figure in

    sns.swarmplot(data=iris, x="species", y=cat, ax=axarr[1])

    sns.boxplot(data=iris, x="species", y=cat, ax=axarr[2])
