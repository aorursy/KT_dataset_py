# Imporitng the required libraries and packages as per below

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir('../input'))
# Naming the colomn feature/attributes as per below and assinging it to 'index' 

index = ['Age_of_the_person', 'Year_of_the_operation', 'No_of_Axillary_nodes_detected_in_total', 'Survival_status_of_person']
# Loading the data set into a pandas dataFrame and assinging it to 'HS'

HS = pd.read_csv('../input/haberman.csv',names = index, header = 1 )

# Exploring the information about the dataset 

HS.info()
HS['Survival_status_of_person'].value_counts()
# printing the first five rows of data set through head fun

HS.head()
# Geting the max,min,std,mean values of each attributes

HS.describe().transpose()

# Performing 1 D Aanlysis of dataset using Distribution plots

sns.FacetGrid(HS, hue="Survival_status_of_person", size=5).map(sns.distplot, "Age_of_the_person").add_legend();

sns.FacetGrid(HS, hue="Survival_status_of_person", size=5).map(sns.distplot, "Year_of_the_operation").add_legend();

sns.FacetGrid(HS, hue="Survival_status_of_person", size=5).map(sns.distplot, "No_of_Axillary_nodes_detected_in_total").add_legend();
plt.show();

Survived=HS.loc[HS['Survival_status_of_person']==1]
Not_Survived=HS.loc[HS['Survival_status_of_person']==2]
# Performing 1 D Aanlysis of dataset using CDF

for i in ["Age_of_the_person","Year_of_the_operation","No_of_Axillary_nodes_detected_in_total"]:
    counts, bin_edges = np.histogram(Survived[i], bins=10,density = True)
    pdf = counts/(sum(counts))
    print(pdf);
    print(bin_edges);
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,ls='dotted',label='surv1_pdf')
    plt.plot(bin_edges[1:], cdf,ls='--',label='surv1_cdf')
    plt.xlabel(i)
    plt.legend()
    
    counts, bin_edges = np.histogram(Not_Survived[i], bins=10,density = True)
    pdf = counts/(sum(counts))
    print(pdf);
    print(bin_edges);
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,label='surv2_pdf')
    plt.plot(bin_edges[1:], cdf,label='surv2_cdf')
    plt.xlabel(i)
    plt.legend()
    plt.show();
# Performing Aanlysis of dataset using BOX plot

sns.boxplot(x='Survival_status_of_person',y='Age_of_the_person', data=HS)
plt.show()

sns.boxplot(x='Survival_status_of_person',y='Year_of_the_operation', data=HS)
plt.show()

sns.boxplot(x='Survival_status_of_person',y='No_of_Axillary_nodes_detected_in_total', data=HS)
plt.show()

# Performing Aanlysis of dataset using violin plot

sns.violinplot(x='Survival_status_of_person',y='Age_of_the_person', data=HS)
plt.show()

sns.violinplot(x='Survival_status_of_person',y='Year_of_the_operation', data=HS)
plt.show()

sns.violinplot(x='Survival_status_of_person',y='No_of_Axillary_nodes_detected_in_total', data=HS)
plt.show()

# Performing Aanlysis of dataset using Pair-Plot

plt.close();
sns.set_style("whitegrid");
sns.pairplot(HS, hue="Survival_status_of_person", size=3);
plt.show()


