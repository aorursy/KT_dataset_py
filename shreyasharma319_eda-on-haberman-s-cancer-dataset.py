import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np





#Load haberman.csv into a pandas dataFrame. This dataset is on the survival of patients who had undergone surgery for breast cancer.

#Breast Cancer = bc

bc_df = pd.read_csv("../input/haberman.csv")



bc_df.shape   #306 rows, 4 columns
bc_df.columns 
bc_df.head()  #the information stored in data
bc_df.info()  # Checking the number of entries and their data type 
bc_df.describe()  
print('Survival Status : ')

x = ((bc_df['status'].value_counts(normalize = True))*100).tolist();

print('Patients surviving for 5 years or longer :'+str(round(x[0],2))+'%');

print('Patients died within 5 years :'+str(round(x[1],2))+'%');
sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "nodes").add_legend();

plt.suptitle('Distribution plot for nodes', size=15);

plt.show();  #due to outliers



sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "age").add_legend();

plt.suptitle('Distribution plot for age', size=15);

plt.show();



sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "year").add_legend();

plt.suptitle('Distribution plot for year', size=15);

plt.show();
#Segregating data on the basis of survival status of patient

bc_status_1 = bc_df.loc[bc_df["status"] == 1]

bc_status_2 = bc_df.loc[bc_df["status"] == 2]
#Creating 3 plots in one cell

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))



#Calculating pdf/cdf for nodes 

counts, bin_edges1 = np.histogram(bc_status_1['nodes'], bins=10, density = True)

pdf1 = counts/(sum(counts))

cdf1 = np.cumsum(pdf1)

ax1.plot(bin_edges1[1:],pdf1, label="pdf_status_1");

ax1.plot(bin_edges1[1:], cdf1, label="cdf_status_1");

ax1.legend()



counts, bin_edges2 = np.histogram(bc_status_2['nodes'], bins=10, density = True)

pdf2 = counts/(sum(counts))

cdf2 = np.cumsum(pdf2)

ax1.plot(bin_edges2[1:],pdf2, label="pdf_status_2");

ax1.plot(bin_edges2[1:], cdf2, label="cdf_status_2")

ax1.legend()



ax1.set(xlabel="pdf/cdf for nodes");

ax1.set_title("PDF/CDF for nodes")



print("*******Feature : Node*******");

print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))

print("\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))





###############################################



#Calculating pdf/cdf for age 

counts, bin_edges1 = np.histogram(bc_status_1['age'], bins=10, density = True)

pdf1 = counts/(sum(counts))

cdf1 = np.cumsum(pdf1)

ax2.plot(bin_edges1[1:],pdf1, label="pdf_status_1");

ax2.plot(bin_edges1[1:], cdf1, label="cdf_status_1");

ax2.legend()



counts, bin_edges2 = np.histogram(bc_status_2['age'], bins=10, density = True)

pdf2 = counts/(sum(counts))

cdf2 = np.cumsum(pdf2)

ax2.plot(bin_edges2[1:],pdf2, label="pdf_status_2");

ax2.plot(bin_edges2[1:], cdf2, label="cdf_status_2")

ax2.legend()



ax2.set(xlabel="pdf/cdf for age");

ax2.set_title("PDF/CDF for age")



print("\n*******Feature : Age*******");

print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))

print("\n\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))





###############################################



#Calculating pdf/cdf for year 

counts, bin_edges1 = np.histogram(bc_status_1['year'], bins=10, density = True)

pdf1 = counts/(sum(counts))

cdf1 = np.cumsum(pdf1)

ax3.plot(bin_edges1[1:],pdf1, label="pdf_status_1");

ax3.plot(bin_edges1[1:], cdf1, label="cdf_status_1");

ax3.legend()



counts, bin_edges2 = np.histogram(bc_status_2['year'], bins=10, density = True)

pdf2 = counts/(sum(counts))

cdf2 = np.cumsum(pdf2)

ax3.plot(bin_edges2[1:],pdf2, label="pdf_status_2");

ax3.plot(bin_edges2[1:], cdf2, label="cdf_status_2")

ax3.legend()



ax3.set(xlabel="pdf/cdf for Year");

ax3.set_title("PDF/CDF for Year")



print("\n*******Feature : Year*******");

print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))

print("\n\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))

i=0;

fig, ax = plt.subplots(1, 3, figsize=(15,5)) #Creating 3 plots in one cell



for feature in bc_df:

    sns.boxplot(x='status',y=feature, data=bc_df, ax=ax[i])

    i= i+1;

    if i == 3:

        break;

ax[0].set_title('Age vs Survival status')

ax[1].set_title('Year vs Survival status')

ax[2].set_title('Nodes vs Survival status')



plt.show()
i=0;

fig, ax = plt.subplots(1, 3, figsize=(15,5)) #Creating 3 plots in one cell



for feature in bc_df:

    sns.violinplot(x='status',y=feature, data=bc_df, ax=ax[i])

    i= i+1;

    if i == 3:

        break;

ax[0].set_title('Age vs Survival status')

ax[1].set_title('Year vs Survival status')

ax[2].set_title('Nodes vs Survival status')



plt.show()

# pairwise scatter plot: Pair-Plot

#This plot establishes relation between 2 features(all possible combinations) in a data frame



sns.set_style("whitegrid");

sns.pairplot(bc_df, hue="status", vars=['year','age','nodes'], height=4) ;

plt.show();

# The diagnol elements are PDFs for each feature. 
## MultiVariate

#2D Density plot, contors-plot





sns.jointplot(x="status", y="nodes", data=bc_df, kind="kde");

plt.show();
