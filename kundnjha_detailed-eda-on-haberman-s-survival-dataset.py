import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



#Load haberman.csv into a pandas dataFrame.

haberman=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",names=['age', 'operation_year', 'axil_nodes', 'survival_status'])


haberman.head(10)
haberman.shape
haberman.columns
haberman.info()
haberman['survival_status'].value_counts()
# Class Label "survival_status" are now labelled as yes & no, stating "yes" as survived and "no" as Not Survived.
haberman['survival_status'] = haberman['survival_status'].map({1:"yes", 2:"no"})
#Updated survival_status
haberman["survival_status"].value_counts()
print(haberman["survival_status"].unique())
print(haberman.groupby("survival_status").count())
status_yes=haberman.loc[haberman["survival_status"]=="yes"]
status_no=haberman.loc[haberman["survival_status"]=="no"]

print("SURVIVAL STATUS : YES -> STATISTICS :")
print(status_yes.describe())
print("\n****************************************************************************\n")
print("SURVIVAL STATUS : NO -> STATISTICS :")
print(status_no.describe())
print("MEDIANS :\n")
print("Median age of the people who survived : ",np.median(status_yes["age"]))
print("Median age of the people who could not survive : ", np.median(status_no["age"]))
print("Median Positive axillary nodes in the people who survived : ", np.median(status_yes["axil_nodes"]))
print("Median Positive axillary nodes in the people who could not survive :  ", np.median(status_no["axil_nodes"]))

print("\n************************************************************************************************\n")

print("QUANTILES :\n")
print("Survival status : Yes")
print("AGE :",np.percentile(status_yes["age"], np.arange(0, 100, 25)))
print("NODES : ", np.percentile(status_yes["axil_nodes"], np.arange(0,100,25)))
print("Survival Status : No")
print("AGE :",np.percentile(status_no["age"], np.arange(0, 100, 25)))
print("NODES : ", np.percentile(status_no["axil_nodes"], np.arange(0,100,25)))

print("\n************************************************************************************************\n")

from statsmodels import robust
print("MEDIAN ABSOLUTE DEVIATION :\n")
print("Survival Status : Yes")
print("AGE :",robust.mad(status_yes["age"]))
print("NODES :",robust.mad(status_yes["axil_nodes"]))
print("Survival Status : No")
print("AGE :",robust.mad(status_no["age"]))
print("NODES :",robust.mad(status_no["axil_nodes"]))

#Analysis of Patient Age
sns.set_style("whitegrid");
sns.FacetGrid(haberman,hue="survival_status",height=6)\
    .map(sns.distplot,"age")\
    .add_legend();

plt.title('Histogram of ages of patients', fontsize=17)
plt.show();
#Analysis of Operation year
sns.FacetGrid(haberman,hue="survival_status",height=6)\
    .map(sns.distplot,"operation_year")\
    .add_legend();

plt.title('Histogram of operation year of patients', fontsize=17)
plt.show();
#Analysis of auxillary nodes
sns.FacetGrid(haberman,hue="survival_status",height=6)\
    .map(sns.distplot,"axil_nodes")\
    .add_legend();

plt.title('Histogram of auxillary nodes detected', fontsize=17)
plt.show();
plt.figure(figsize=(20,6))
plt.subplot(131)
counts,bin_edges=np.histogram(status_yes["age"],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,linewidth=3.0)
plt.plot(bin_edges[1:],cdf,linewidth=3.0)
plt.xlabel('AGE')
plt.ylabel("COUNT")
plt.title('PDF-CDF of AGE for Survival Status = YES')
plt.legend(['PDF-AGE', 'CDF-AGE'], loc = 5,prop={'size': 12})

plt.subplot(132)
counts,bin_edges=np.histogram(status_yes["operation_year"],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,linewidth=3.0)
plt.plot(bin_edges[1:],cdf,linewidth=3.0)
plt.xlabel('YEAR OF OPERATION')
plt.ylabel("COUNT")
plt.title('PDF-CDF of OPERATION YEAR for Survival Status = YES')
plt.legend(['PDF-OPERATION YEAR', 'CDF-OPERATION YEAR'], loc = 5,prop={'size': 11})

plt.subplot(133)
counts,bin_edges=np.histogram(status_yes["axil_nodes"],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,linewidth=3.0)
plt.plot(bin_edges[1:],cdf,linewidth=3.0)
plt.xlabel('AUXILLARY NODES')
plt.ylabel("COUNT")
plt.title('PDF-CDF of AXIL NODES for Survival Status = YES')
plt.legend(['PDF-AXIL NODES', 'CDF-AXIL NODES'], loc = 5,prop={'size': 11})
plt.show()

sns.boxplot(x = "survival_status" , y = "age", data = haberman )
plt.title("1. Box plot for survival_status and Age")
plt.show()

sns.boxplot(x = "survival_status" , y = "operation_year", data = haberman )
plt.title("\n2. Box plot for survival_status and Operation Year")
plt.show()

sns.boxplot(x = 'survival_status', y = 'axil_nodes', data = haberman)
plt.title("\n3. Box plot for survival_status and Auxillary Nodes")
plt.show()
sns.violinplot(x = 'survival_status', y = 'age', data = haberman)
plt.title("Violin plot for survival_status and Age")
plt.show()

sns.violinplot(x = 'survival_status', y = 'operation_year', data = haberman)
plt.title("\nViolin plot for survival_status and Operation Year")
plt.show()

sns.violinplot(x = 'survival_status', y = 'axil_nodes', data = haberman)
plt.title("\nViolin plot for survival_status and Auxillary Node")
plt.show()
sns.jointplot(x="age",y="operation_year",data=haberman, kind="kde")
plt.show()

sns.jointplot(x="age",y="axil_nodes",data=haberman, kind="kde")
plt.show()

sns.jointplot(x="operation_year",y="axil_nodes",data=haberman, kind="kde")
plt.show()
# AGE VS AUXILLARY NODES
sns.FacetGrid(haberman, hue="survival_status", height=6) \
   .map(plt.scatter, "age", "axil_nodes") \
   .add_legend();
plt.show();
#AUXILLARY NODES VS OPERATION YEAR
sns.FacetGrid(haberman, hue="survival_status", height=6) \
   .map(plt.scatter, "axil_nodes", "operation_year") \
   .add_legend();

plt.show();
#AGE VS OPERATION YEAR
sns.FacetGrid(haberman, hue="survival_status", height=6) \
   .map(plt.scatter, "operation_year", "age") \
   .add_legend();
plt.show();
plt.close()
sns.pairplot(haberman,hue="survival_status",height=3.5)
plt.show()