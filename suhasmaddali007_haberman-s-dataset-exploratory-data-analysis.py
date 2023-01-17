import numpy as np                         #importing the numpy library for scientific calculation

import seaborn as sns                      #importing seaborn for visualization 

import matplotlib.pyplot as plt            #This is also used for plotting 

import pandas as pd                        #This is used to create data frames and also read and write files

%matplotlib inline                        
haberman = pd.read_csv('../input/haberman.csv')       #data is read from the current directory
haberman.shape                           #Looking at the current shape of the dataset under consideration
haberman.head()                          #Looking at various columns and how the data looks like 
haberman['status'].value_counts()                  #counting the number of output labels and their distribution
haberman.plot(x = 'age', y = 'operation_year', kind = 'scatter')            #Using a scatter plot to see the distribution
haberman.plot(x = 'operation_year', y = 'axil_nodes', kind = 'scatter')         #Using various other features to check the scatter plot
sns.pairplot(data = haberman, hue = 'status', size = 3)                 #Using pairplot to see the relationship with all the input features

plt.show()
sns.set_style(style = "whitegrid")

sns.FacetGrid(haberman, hue = "status", size = 5).map(plt.scatter, 'axil_nodes','operation_year').add_legend()
#dividing the data into 2 parts namely haberman_1 and haberman_2

haberman_1 = haberman.loc[haberman['status'] == 1]      #storing the values of the input where the status is 1               

haberman_2 = haberman.loc[haberman['status'] == 2]      #storing the values of the input where the status is 2

#Checking how the age is distributed and comparing between status 1 and status 2

plt.plot(haberman_1["age"], np.zeros_like(haberman_1['age']), 'o')   

plt.plot(haberman_2["age"], np.zeros_like(haberman_2['age']), 'o')
haberman.head(1)         #Having a look at the column just to access it later in the next cell
sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'age').add_legend()      #Having a look at the distribution how the age is distributed and trying to see the co-relation between age and status
sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'operation_year').add_legend()    #Checking the co-relation between year and status
haberman.head(1)
sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'axil_nodes').add_legend()       #Checking the corelation between nodes and status
counts, bin_edges = np.histogram(haberman['axil_nodes'], density = True)       #Creating the values for the histogram

pdf = counts / sum(counts)                                                #Computing the pdf of the above values

cdf = np.cumsum(pdf)                                                      #Computing the cdf based on the pdf calculated above      

print(bin_edges)                                                          #Printing the edges of the histogram

plt.plot(bin_edges[1:], pdf)                                              #Plotting the pdf of the above 

plt.plot(bin_edges[1:], cdf)                                              #Plotting the cdf of the above

counts, bin_edges = np.histogram(haberman['axil_nodes'], density = True, bins = 20) #Creating the values of the histogram for haberman_2

pdf = counts / sum(counts)                                                #Creating the pdf of haberman_2

plt.plot(bin_edges[1:], pdf)                                              #Plotting the pdf of the haberman_2
#Repeating the steps in the above cells with slight modification

counts1, bin_edges = np.histogram(haberman_1['axil_nodes'], density = True)

pdf1 = counts1 / sum(counts1)

cdf1 = np.cumsum(pdf1)

counts2, bin_edges = np.histogram(haberman_2['axil_nodes'], density = True)

pdf2 = counts2 / sum(counts2)

cdf2 = np.cumsum(pdf2)

plt.plot(bin_edges[1:], pdf1)

plt.plot(bin_edges[1:], cdf1)

plt.plot(bin_edges[1:], pdf2)

plt.plot(bin_edges[1:], cdf2)
print('The mean of the nodes of haberman dataset which has status 1 and status 2 are given below')

print(np.mean(haberman_1['axil_nodes']))         #calculating the mean of the nodes with status 1

print(np.mean(haberman_2['axil_nodes']))         #calculating the mean of the nodes with status 2

print('The standard deviation of the nodes of haberman dataset which has status 1 and status 2 are given below')

print(np.std(haberman_1['axil_nodes']))          #calculating the standard deviation with status 1

print(np.std(haberman_2['axil_nodes']))          #calculating the standard deviation with status 2
sns.distplot(haberman_1['axil_nodes'])           #Plotting the distribution of nodes with status = 1
sns.distplot(haberman_2['axil_nodes'], color = 'green')    #Plotting the distribution of nodes with status = 2
print('The 0, 25 and 75th percentiles of the given nodes of haberman dataset with status 1 and status 2 is')

print(np.percentile(haberman_1['axil_nodes'], np.arange(0, 100, 25)))       #printing the 0th, 25th, 50th and 75th percentile of nodes with status = 1

print(np.percentile(haberman_2['axil_nodes'], np.arange(0, 100, 25)))       #printing the 0th, 25th, 50th and 75th percentile of nodes with status = 2
print('the 95th percentile of the nodes that are present in haberman data set with status 1 and 2 are given below')

print(np.percentile(haberman_1['axil_nodes'], 95))     #printing the 95th percentile of nodes with status = 1

print(np.percentile(haberman_2['axil_nodes'], 95))     #printing the 95th percentile of nodes with status = 2
print('This is a box plot taking nodes as the feature and seperating them based on hue = status')

sns.boxplot(x = 'status', y = 'axil_nodes', data = haberman)    #Plotting the box plot for x as status and y as nodes
print('Let us also use the violen plot that would also take into account how the points are spread')

sns.violinplot(x = 'status', y = 'axil_nodes', data = haberman)    #also plotting the violin plot for x as status and y as nodes