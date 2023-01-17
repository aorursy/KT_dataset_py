import numpy as np
import seaborn as sns
import pandas as pd

#reading data from csv and adding column names(as the data do not contain the columns)
habermans=pd.read_csv("../input/haberman.csv",
                      names=["age_patients","year_operation","positive_axillary_nodes","survival_status"])
habermans
#High Level statistics on the data
print("Number of datapoints in the Haberman's dataset ",habermans.shape[0])
print("Number of features in the Haberman's dataset ",habermans.shape[1]-1)#because last column is class label
print("Count of each class in habermans dataset \n",habermans["survival_status"].value_counts())
import matplotlib.pyplot as plt
sns.FacetGrid(hue="survival_status",data=habermans,height=4).map(sns.distplot,"age_patients").add_legend()
plt.title("Histogram's/Pdf for patients age")
plt.show()
#creating CDF plot
age_survival1=habermans[habermans["survival_status"]==1].age_patients
age_survival2=habermans[habermans["survival_status"]==2].age_patients

count_survival1,bin_survival1=np.histogram(age_survival1,bins=10,density=True)
count_survival2,bin_survival2=np.histogram(age_survival2,bins=10,density=True)

#calculating Pdf
pdf1=(count_survival1/sum(count_survival1))
pdf2=count_survival2/sum(count_survival2)

#calculating cdf
cdf1=np.cumsum(pdf1)
cdf2=np.cumsum(pdf2)

#plotting
plt.plot(bin_survival1[1:],pdf1,label="pdf_survivalStatus_1")
plt.plot(bin_survival1[1:],cdf1,label="cdf_survivalStatus_1")

plt.plot(bin_survival2[1:],pdf2,label="pdf_survivalStatus_2")
plt.plot(bin_survival2[1:],cdf2,label="cdf_survivalStatus_2")

plt.xlabel("Age of patients")
plt.ylabel("Probablity")
plt.title("CDF of age_patients")
plt.legend()
plt.show()
#drawing box plot and violin plot to understand more

plt.close()
plt.title("Box plot for age_patients")
sns.boxplot(x="survival_status",y="age_patients",data=habermans)
plt.show()
plt.close()
plt.title("violin plot for age_patients")
sns.violinplot(x="survival_status",y="age_patients",data=habermans)
plt.show()
sns.FacetGrid(habermans,hue="survival_status",height=4).map(sns.distplot,"year_operation").add_legend()
plt.title("Histogram/pdf's for year_operation")
plt.show()

plt.close()
plt.title("Box plot for year_operation")
sns.boxplot(x="survival_status",y="year_operation",data=habermans)
plt.show()

plt.close()
plt.title("Violin plot for year_operation")
sns.violinplot(y="year_operation",x="survival_status",data=habermans)
plt.show()
#CDF for year_operation
survivalStatus_1=habermans[habermans["survival_status"]==1]
survivalStatus_2=habermans[habermans["survival_status"]==2]

count1_yr,bin1_year=np.histogram(survivalStatus_1.year_operation,bins=10)
count2_yr,bin2_year=np.histogram(survivalStatus_2.year_operation,bins=10)

pdf1=count1_yr/sum(count1_yr)
pdf2=count2_yr/sum(count2_yr)

cdf1=np.cumsum(pdf1)
cdf2=np.cumsum(pdf2)

plt.plot(bin1_year[1:],pdf1)
plt.plot(bin1_year[1:],cdf1)
plt.plot(bin2_year[1:],pdf2)
plt.plot(bin2_year[1:],cdf2)

plt.xlabel("year of operation")
plt.ylabel("probability")
plt.title("CDF for year_operation")
plt.show()
sns.FacetGrid(hue="survival_status",height=4,data=habermans).map(sns.distplot,"positive_axillary_nodes").add_legend()
plt.show()
#CDF plot
count_node,bin_node=np.histogram(survivalStatus_1.positive_axillary_nodes,bins=10)
pdf_node=count_node/sum(count_node)
cdf_node=np.cumsum(pdf_node)

plt.plot(bin_node[1:],pdf_node)
plt.plot(bin_node[1:],cdf_node)
plt.xlabel("Positive Auxillary Nodes")
plt.ylabel("Probability")
plt.title("CDF for positive_axillary_nodes")
plt.show()
#plotting box plot 
plt.close()
sns.boxplot(x="survival_status",y="positive_axillary_nodes",data=habermans)
plt.show()
#violin plot
plt.close()
sns.violinplot(x="survival_status",y="positive_axillary_nodes",data=habermans)
plt.show()
#building 2D scatter plot to understand the relationship between patients age and positive axillary nodes to determine
#survival rate
sns.FacetGrid(habermans,hue="survival_status",height=4).map(plt.scatter,"age_patients","positive_axillary_nodes").add_legend()
plt.show()
sns.FacetGrid(habermans,hue="survival_status",height=4).map(plt.scatter,"age_patients","year_operation").add_legend()
plt.show()
sns.FacetGrid(habermans,hue="survival_status",height=4).map(plt.scatter,"year_operation","positive_axillary_nodes").add_legend()
plt.show()
#Let us draw pair plot to understand more
plt.close();
sns.set_style("whitegrid");
sns.pairplot(habermans, hue="survival_status", vars=["age_patients","year_operation","positive_axillary_nodes"],height=3);
plt.show()
