import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



#Load haberman.csv into pandas dataframe

haberman=pd.read_csv("../input/haberman.csv")

print(haberman)

#Shape of data,ie no: of datapoints in haberman datset



print(haberman.shape)
#Labels of columns present in haberman dataset



print(haberman.columns)
#Datapoints belongs to each status



print(haberman["status"].value_counts())
#2-D scatter plot:





haberman.plot(kind='scatter', x='nodes', y='age') ;

plt.show()



#2-D scatter plot with color coding for each type of datapoints



sns.set_style("darkgrid")

sns.FacetGrid(haberman,hue="status",size=4).map(plt.scatter,"nodes","age").add_legend()

plt.show()
#Pair plot is a smart way to visualise plots between all features of data set as pairs

plt.close();

sns.set_style("darkgrid");

sns.pairplot(haberman,hue="status",size=3,vars=['age','year','nodes'])

plt.show()
#1D Scatter plot-Plotting of each of the variables by seeting y axis as zero and x axis the intended feature for each class- label

#Below is the 1D scatter plot on nodes feature



haberman_one=haberman.loc[haberman["status"]==1]

haberman_two=haberman.loc[haberman["status"]==2]



#Plotting

plt.plot(haberman_one["nodes"], np.zeros_like(haberman_one['nodes']), 'o')

plt.plot(haberman_two["nodes"], np.zeros_like(haberman_two['nodes']), 'o')

plt.show()
#pdf is the number of data point situated at a particular region 



sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'nodes').add_legend()

plt.show()
sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'age').add_legend()

sns.FacetGrid(haberman,hue='status',size=5).map(sns.distplot,'year').add_legend()

plt.show()
# CDF helps in calculating the percentage of people survived after surgery



#Below code evaluates cdf of patients surviving more,that is survival status one



counts, bin_edges = np.histogram(haberman_one['nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



plt.show();




#Below code evaluates cdf of patients surviving less,that is survival status two



counts, bin_edges = np.histogram(haberman_two['nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF plotting together for more survival and less survival chances

counts, bin_edges = np.histogram(haberman_one['nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haberman_two['nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

plt.show();

# Mean is the average of all data points

# Variance is the summation of  all data points deviation from its mean,that is spread.if spread is less data point will be closer to the mean 

# Standard deviation is the square root of mean

print("Means:")

print(np.mean(haberman_one["nodes"]))

#Mean with an outlier.

print(np.mean(np.append(haberman_one["nodes"],50)));

print(np.mean(haberman_two["nodes"]))





print("\nStd-dev:");

print(np.std(haberman_one["nodes"]))

print(np.std(haberman_two["nodes"]))

#Median, Quantiles, Percentiles, IQR.

print("\nMedians:")

print(np.median(haberman_one["nodes"]))

#Median with an outlier

print(np.median(np.append(haberman_one["nodes"],50)));

print(np.median(haberman_two["nodes"]))







print("\nQuantiles:")

print(np.percentile(haberman_one["nodes"],np.arange(0, 100, 25)))

print(np.percentile(haberman_two["nodes"],np.arange(0, 100, 25)))



print("\n90th Percentiles:")

print(np.percentile(haberman_one["nodes"],90))

print(np.percentile(haberman_two["nodes"],90))





from statsmodels import robust

print ("\nMedian Absolute Deviation")

print(robust.mad(haberman_one["nodes"]))

print(robust.mad(haberman_two["nodes"]))



#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitively.



sns.boxplot(x='status',y='nodes', data=haberman)

plt.show()
# A violin plot combines the benefits of the previous two plots 

#and simplifies them



# Denser regions of the data are fatter, and sparser ones thinner 

#in a violin plot



sns.violinplot(x='status',y='nodes', data=haberman, size=8)

plt.show()
#Contour plot is a method of visualizing the  2-D scatter plot more intuitively

sns.jointplot(x="age", y="nodes", data=haberman_one, kind="kde");

plt.show();