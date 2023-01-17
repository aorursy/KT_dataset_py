import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np
tips = sns.load_dataset('tips')

tips.head()
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.mean)
import numpy as np
plt.figure(figsize=(10,8))

sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
sns.countplot(x='sex',data=tips)
plt.figure(figsize=(10,8))

sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')
# Can do entire dataframe with orient='h'

plt.figure(figsize=(12,8))

sns.boxplot(data=tips,palette='rainbow',orient='h')
plt.figure(figsize=(16,8))

sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")
plt.figure(figsize=(16,8))

sns.violinplot(x="day", y="total_bill", data=tips,palette='rainbow')
plt.figure(figsize=(16,8))

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',palette='Set1')
plt.figure(figsize=(16,8))

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')
plt.figure(figsize=(9,6))

sns.stripplot(x="day", y="total_bill", data=tips)
plt.figure(figsize=(11,8))

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True)
plt.figure(figsize=(10,8))

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1')


plt.figure(figsize=(12,8))

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True)
plt.figure(figsize=(12,8))

sns.swarmplot(x="day", y="total_bill", data=tips)
plt.figure(figsize=(12,8))

sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True)
plt.figure(figsize=(12,8))

sns.violinplot(x="tip", y="day", data=tips,palette='rainbow')

sns.swarmplot(x="tip", y="day", data=tips,color='black',size=3)
sns.factorplot(x='sex',y='total_bill',data=tips,kind='bar')
plt.figure(figsize=(12,8))

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
#To get the kde plot we can sum these basis functions.



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
# Just the Grid

g = sns.FacetGrid(tips, col="time", row="smoker")
g = sns.FacetGrid(tips, col="time",  row="smoker")

g = g.map(plt.hist, "total_bill")
g = sns.FacetGrid(tips, col="time",  row="smoker")

g = g.map(plt.hist, "total_bill")
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = sns.JointGrid(x="total_bill", y="tip", data=tips)

g = g.plot(sns.regplot, sns.distplot)
flights = sns.load_dataset('flights')

flights.head()
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