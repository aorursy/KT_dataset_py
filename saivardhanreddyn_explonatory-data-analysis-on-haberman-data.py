import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
haberman=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",names=['age', 'year', 'nodes', 'status'])
#(Q) How many data points are present and features ?
print(haberman.shape)
#(Q) what are the column names or features ?
print(haberman.columns)

#(Q) How many data points for each class are present?
#(Q) this is a imbalanced data set
print(haberman['status'].value_counts())
#(Q) Number of classes ?
print(haberman['status'].nunique())
#describing the dataframe
print(haberman.describe())
#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.
haberman.plot(kind='scatter', x='age', y='year') ;
plt.title("2-D PLOT BETWEEN AGE AND YEAR")
plt.show()
#THERE IS A PLOT BETWEEN AGE AND YEAR WE WONT BE GETTING MUCH INFORMATION AS WE DIDN'T CLASSIFY BASED ON STATUS
#WE NEED TO COLOR CODE BASED ON STATUS
#here instead of getting confused by using status as 1 ans 2 we can use yes or no
haberman['status']=haberman['status'].apply(lambda x:'Survived' if x==1 else 'Didntsurvive')
# 2-D Scatter plot with color-coding for each patient.
plt.close()
sns.set_style("whitegrid");
sns.FacetGrid(haberman,hue='status',height=4)\
.map(plt.scatter,"age","year")\
.add_legend();
plt.title("2-D Scatter plot with color-coding for each patient")
plt.show();      
# here the graph is between age and year
# here the data points are overlapping
# notice that the orange and blue points can't be seperated easily
#there line can't be drawn between the orange annd blue
#(Q) How many combinations are present 3c2 
#3d plot
import plotly.express as px
df = pd.read_csv("../input/habermans-survival-data-set/haberman.csv",names=['age', 'year', 'nodes', 'status'])
fig = px.scatter_3d(df, x='age', y='year', z='nodes',
              color='status')
plt.title("3D PLOT BETWEEN AGE AND YEAR AND NODES WITH COLOR AS STATUS")
fig.show()
# (PAIR PLOT)
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman,hue="status",height=3)
plt.title("PAIR PLOT")
plt.show()

#1d scatter plot using one feature
import numpy as np
haberman_survived=haberman.loc[haberman['status']=='Survived']
haberman_Didntsurvive=haberman.loc[haberman['status']=='Didntsurvive']
plt.plot(haberman_survived['nodes'],np.zeros_like(haberman_survived['nodes']),'o')
plt.plot(haberman_Didntsurvive['nodes'],np.zeros_like(haberman_Didntsurvive['nodes']),'o')
plt.title("1D SCATTER PLOT FOR NODES FOR THOSE WHO SURVIVED AND DIDNT SURVIVE")
plt.show()
sns.FacetGrid(haberman, hue="status", height=5) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.title("PDF of Nodes")
plt.show();

sns.FacetGrid(haberman, hue="status", height=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.title("PDF of year")
plt.show();

sns.FacetGrid(haberman, hue="status", height=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.title("PDF FOR AGE" )
plt.show();

#Histograms and probability density functions
counts,bins=np.histogram(haberman_survived['nodes'],bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)
print(bins)
cdf=np.cumsum(pdf)
plt.plot(bins[1:],pdf,label='pdf')
plt.plot(bins[1:],cdf,label='cdf')
plt.xlabel('number of nodes')
plt.ylabel('probability')
plt.title("CDF AND PDF OF NODES FOR SURVIVED")
plt.legend()
plt.show()
counts,bins=np.histogram(haberman_survived['age'],bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)
print(bins)
cdf=np.cumsum(pdf)
plt.plot(bins[1:],pdf,label='pdf')
plt.plot(bins[1:],cdf,label='cdf')
plt.xlabel('age')
plt.ylabel('probability')
plt.title("CDF AND PDF FOR THE AGE OF SURVIVED")
plt.legend()
plt.show()
counts,bins=np.histogram(haberman_survived['year'],bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)
print(bins)
cdf=np.cumsum(pdf)
plt.plot(bins[1:],pdf,label='pdf')
plt.plot(bins[1:],cdf,label='cdf')
plt.xlabel('year')
plt.ylabel('probability')
plt.title("CDF AND PDF FOR THE YEAR OF OPERATION FOR THE SURVIVED")
plt.legend()
plt.show()
counts,bins=np.histogram(haberman_Didntsurvive['nodes'],bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)
print(bins)
cdf=np.cumsum(pdf)
plt.plot(bins[1:],pdf,label='pdf')
plt.plot(bins[1:],cdf,label='cdf')
plt.xlabel('nodes')
plt.ylabel('probability')
plt.title("CDF AND PDF OF THE NODES FOR THE DIDNT SURVIVE")
plt.legend()
plt.show()
# Plots of CDF of nodes for patients .
#survive
counts, bin_edges = np.histogram(haberman_survived['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# notsurvive
counts, bin_edges = np.histogram(haberman_Didntsurvive['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.title("BOTH CDF AND PDF OF SURVIVE AND DIDNT SURVIVE")
plt.show()
print("Means of nodes of survived and didnt survive:")
print(np.mean(haberman_survived["nodes"]))
print(np.mean(haberman_Didntsurvive["nodes"]))
print("Means of age of survived and didnt survive:")
print(np.mean(haberman_survived["age"]))
print(np.mean(haberman_Didntsurvive["age"]))
print("Means of year of operation of survived and didnt survive:")
print(np.mean(haberman_survived["year"]))
print(np.mean(haberman_Didntsurvive["year"]))
print("standard deviation of nodes of survived and didnt survive:")
print(np.std(haberman_survived["nodes"]))
print(np.std(haberman_Didntsurvive["nodes"]))
print("standard deviation of age of survived and didnt survive:")
print(np.std(haberman_survived["age"]))
print(np.std(haberman_Didntsurvive["age"]))
print("standard deviation of year of operation of survived and didnt survive:")
print(np.std(haberman_survived["year"]))
print(np.std(haberman_Didntsurvive["year"]))
#we are not using mean as an outlier may change the value of mean
#here i can see the significant difference between nodes mean and std
#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(haberman_survived["nodes"]))
print(np.median(haberman_Didntsurvive["nodes"]))
print("\nQuantiles:")
print(np.percentile(haberman_survived["nodes"],np.arange(0, 101, 25)))
print(np.percentile(haberman_Didntsurvive["nodes"],np.arange(0, 101, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman_survived["nodes"],90))
print(np.percentile(haberman_Didntsurvive["nodes"],90))
from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_survived["nodes"]))
print(robust.mad(haberman_Didntsurvive["nodes"]))
print("\nQuantiles of whole:")
print(np.percentile(haberman["nodes"],np.arange(0, 101, 25)))

sns.boxplot(x='status',y='nodes', data=haberman)
plt.title("BOX PLOT AND WHISKERS FOR SURVIVE AND DIDNT SURVIVE")
plt.show()
# no of people who have survived we can see that they have highly densed 0-5 nodes
# 75% who didntsurvive had node count less than 12

sns.violinplot(x="status", y="nodes", data=haberman, size=8)
plt.title("VIOLIN PLOT FOR SURVIVE AND DIDNT SURVIVE")
plt.show()
print(haberman_survived[haberman_survived["nodes"]==0].shape)

print("the person who survived  with 0 nodes are given by 117 of 306 which is "+(str(round(117/306*100))+"%"))

# the patients who didn't survive more than 5 years having 0 nodes are
print(haberman_Didntsurvive[haberman_Didntsurvive["nodes"]==0].shape)
print("the person who did not survived  with  0 nodes are given by 117 of 306 which is "+(str(round(19/306*100))+"%"))
#"the pateints who survived with 1-3 nodes are given by")
s=haberman_survived[haberman_survived["nodes"]>=1]
print(s[s['nodes']<=3].shape)

print("the persons who got survived with 1-3 nodes are given by 61 0f 306 which is "+str(round(61/306*100))+"%")
s=haberman_Didntsurvive[haberman_Didntsurvive["nodes"]>=1]
print(s[s['nodes']<=3].shape)

print("the persons who got survived with 1-3 nodes are given by 61 0f 306 which is "+str(round(20/306*100))+"%")
haberman_survival1=haberman.loc[haberman["status"]=="Survived"]
sns.jointplot(x="age",y="nodes",data=haberman_survival1,kind="kde")
plt.title("CONTOUR PLOT BETWEEN NODES AND AGE")
plt.grid()
plt.show()