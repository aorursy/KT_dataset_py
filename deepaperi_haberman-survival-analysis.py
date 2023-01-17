# Reading the haberman csv data set

import pandas 
import seaborn
import matplotlib.pyplot as plt
import numpy 
#Loading Iris.csv into a pandas dataFrame.
cancer = pandas.read_csv("../input/haberman.csv")
# (Q) how many data-points and features?
# (or)size of the matrix of dataset
print(cancer.shape)
#(Q) What are the column names in our dataset?
print(cancer.columns)
#(Q) How many data points for each class are present? 
# or how many people survived for 5 or more than 5 years and how many died
cancer["Surv_status"].value_counts()

# Printing the top 5 rows for inital analysis of the dataset and for verfication

cancer.head()
# total number of observations
print(cancer.info())
# unique constraints
list(cancer['Surv_status'].unique())
# replacing 1 - survived and 2 - died
#cancer["Surv_status"].replace(to_replace = {
#   1: "Survived",
#    2: "Died"
#}, inplace = True)

cancer['Surv_status'] = cancer['Surv_status'].apply({1: 'survived', 2: 'died'}.get)
cancer


# verification of the values whether assigned or not

cancer["Surv_status"].value_counts()
# finding the unique variables 

list(cancer['Surv_status'].unique())
# checking the overview of data set description

cancer.describe()
cancer["Surv_status"].value_counts()
print("\n" + str(cancer["Surv_status"].value_counts(normalize = True)))
# checking if any null values
cancer.isnull().any()
cancer.count()
status1 = cancer.loc[cancer["Surv_status"] == "survived"]

plt.plot(status1["Age"], numpy.zeros_like(status1["Age"]), '2')#blue
plt.plot(status1["Op_Year"], numpy.zeros_like(status1["Op_Year"]), '*')#orange
plt.plot(status1["axial_nodes_det"], numpy.zeros_like(status1["axial_nodes_det"]), '.')#green
plt.show()





status2 = cancer.loc[cancer["Surv_status"] == "died"]

plt.plot(status2["Age"], numpy.zeros_like(status2["Age"]), '2')#blue
plt.plot(status2["Op_Year"], numpy.zeros_like(status2["Op_Year"]), '*')#orange
plt.plot(status2["axial_nodes_det"], numpy.zeros_like(status2["axial_nodes_det"]), '.')#green
plt.show()



seaborn.FacetGrid(cancer, hue = "Surv_status", height = 5)\
        .map(seaborn.distplot, "Age")\
        .add_legend()
plt.show()

seaborn.FacetGrid(cancer, hue = "Surv_status", height = 5)\
        .map(seaborn.distplot, "Op_Year")\
        .add_legend()
plt.show()
seaborn.FacetGrid(cancer, hue = "Surv_status", height = 5)\
        .map(seaborn.distplot, "axial_nodes_det")\
        .add_legend()
plt.show()
counts, bin_edges = numpy.histogram(status1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = numpy.histogram(status1['Age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();

counts, bin_edges = numpy.histogram(status1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();

# blue - pdf
# orange - cdf

counts, bin_edges = numpy.histogram(status1['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


counts, bin_edges = numpy.histogram(status1['axial_nodes_det'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


print("age")
counts, bin_edges = numpy.histogram(status1['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

print("Op_Year")
counts, bin_edges = numpy.histogram(status1['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

print("axial_nodes_det")

counts, bin_edges = numpy.histogram(status1['axial_nodes_det'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = numpy.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();

plt.close()
cancer.hist()
plt.figure(figsize=(18,8))

seaborn.boxplot(x='Surv_status',y='Age', data= cancer)
plt.show()


seaborn.boxplot(x='Surv_status',y='Op_Year', data= cancer)
plt.show()


seaborn.boxplot(x='Surv_status',y='axial_nodes_det', data= cancer)
plt.show()

seaborn.violinplot(x="Surv_status", y="Age", data=cancer, height=8)
plt.show()
seaborn.violinplot(x="Surv_status", y="Op_Year", data=cancer, height=8)
plt.show()
seaborn.violinplot(x="Surv_status", y="axial_nodes_det", data=cancer, height=8)
plt.show()
print(cancer.columns)
cancer.plot(kind='scatter', x='Age', y='Op_Year') ;
plt.show()
seaborn.set_style("whitegrid");
seaborn.FacetGrid(cancer, hue="Surv_status", height=4) \
   .map(plt.scatter, "Age", "Op_Year") \
   .add_legend();
plt.show();
seaborn.set_style("whitegrid");
seaborn.FacetGrid(cancer, hue="Surv_status", height=4) \
   .map(plt.scatter, "Age", "axial_nodes_det") \
   .add_legend();
plt.show();
seaborn.set_style("whitegrid");
seaborn.FacetGrid(cancer, hue="Surv_status", height=4) \
   .map(plt.scatter, "Op_Year", "axial_nodes_det") \
   .add_legend();
plt.show();
seaborn.set_style("whitegrid");
seaborn.FacetGrid(cancer, hue="Surv_status", height=4) \
   .map(plt.scatter, "Age", "Op_Year", "axial_nodes_det") \
   .add_legend();
plt.show();
plt.close();
seaborn.set_style("whitegrid");
seaborn.pairplot(cancer, hue="Surv_status", height=3);
plt.show()
cancer['Surv_status'] = cancer['Surv_status'].apply({'survived': 0, 'died': 1}.get)
cancer
import mpl_toolkits.mplot3d
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(cancer['Age'], cancer['Op_Year'], cancer['axial_nodes_det'], 
           c = cancer['Surv_status'].map(lambda x: {True:'green', False:'red'}[x]), marker = 'o')
ax.set_xlabel('Age')
ax.set_ylabel('Op_Year')
ax.set_zlabel('axial_nodes_det')
plt.show()
cancer['Surv_status'] = cancer['Surv_status'].apply({0: 'survived', 1: 'died'}.get)
cancer
seaborn.jointplot(x="Age", y="axial_nodes_det", data=cancer, kind="kde");
plt.show();
def CancerAnalysis(age, opyear, node):
    
    '''This function returns
        True: if a patient will survive for more than 5 years
        False: otherwise
    '''
    
    if age <= 40:
        return True;
    elif age > 77:
        return False;
    elif age < 77 & age > 70: 
        return True;
    elif node <= 4:
        return True;
    else:
        return False; # analysis is not accurate in this part of else
    
    
