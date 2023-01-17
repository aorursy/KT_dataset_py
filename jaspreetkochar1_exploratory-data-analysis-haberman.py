import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Since haberman.csv is unlabelled, let us assign a name to each of the features.
features=['age','operation_year','axillary_nodes','survival_status']
#Load the Haberman dataset into a pandas dataFrame.
haberman = pd.read_csv("../input/haberman.csv",names=features)
# Let us a have a sneak peak at the dataset. 
haberman.head()
# Data points and the number of features.
print(haberman.shape)
haberman['survival_status'].value_counts()
#Map survival_status to a categorical attribute('Survived','Died')
haberman['survival_status'] = haberman['survival_status'].map({1:'Survived',2:'Died'})
haberman.head()
# 2-D Scatter plot with color-coding to indicate survival status.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="survival_status", size=5) \
   .map(plt.scatter, "age", "axillary_nodes") \
   .add_legend();
plt.show();
# Visualization of multiple 2-D scatter plots for each combination of features. 
# # pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="survival_status",vars=['age','operation_year','axillary_nodes'] ,size=4);
plt.show()
#Collect the data into two survival states i.e. survived and died for univariate analysis.
haberman_survived_data = haberman.loc[haberman['survival_status'] == 'Survived']
haberman_died_data = haberman.loc[haberman['survival_status'] == 'Died']

# Plot histogram,PDF for each feature.
for col in haberman.loc[:,haberman.columns!='survival_status']:
    sns.FacetGrid(haberman, hue="survival_status", size=5).map(sns.distplot, col).add_legend();
    plt.show();
# Plot Cumulative Distribution Function (CDF) for both survived and died patients.
# We can visually see what percentage of survived patients have
# axillary nodes less than 3?

#Plot CDF of both survived and died patients.

for col in haberman.loc[:,haberman.columns!='survival_status']:
    counts, bin_edges = np.histogram(haberman_survived_data[col], bins=20, 
                                 density = True)
    pdf = counts/(sum(counts))
    #compute CDF
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(col)
    
    counts, bin_edges = np.histogram(haberman_died_data[col], bins=20, 
                                 density = True)
    pdf = counts/(sum(counts))
    #compute CDF
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.legend('survival_status')
    plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])
    plt.xlabel(col)
    plt.show();

# Statistics of the entire dataset.
haberman.describe()
# Statistics of the people who survived.
haberman_survived_data.describe()
# Statistics of the people who didn't survive
haberman_died_data.describe()
# Compute the median absolute deviation to gauge the spread of the distributions effectively.
from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_survived_data["axillary_nodes"]))
print(robust.mad(haberman_died_data["axillary_nodes"]))
#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.

#IN the plot below, a technique call inter-quartile range is used in plotting the whiskers. 
#Whiskers in the plot below donot correposnd to the min and max values.

#Box-plot can be visualized as a PDF on the side-ways.

sns.boxplot(x='survival_status',y='axillary_nodes', data=haberman)
plt.show()

sns.boxplot(x='survival_status',y='age', data=haberman)
plt.show()
# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner 
#in a violin plot

sns.violinplot(x='survival_status',y='axillary_nodes', data=haberman)
plt.show()

sns.violinplot(x='survival_status',y='age', data=haberman)
plt.show()

sns.violinplot(x='survival_status',y='operation_year', data=haberman)
plt.show()