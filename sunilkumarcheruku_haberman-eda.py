#IMPORTING Libraries needd for EDA 

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
#Reading the haberman file

haberman=pd.read_csv("../input/haberman.csv")
#Disply column names present in dataset

print(haberman.columns)
#Header is not we have to insert the header

haberman.columns=['age', 'year_of_operation', 'aux_nodes_detected', 'survival_status']
haberman.columns
#To see the format of data

haberman.head()
haberman.tail()
haberman.describe()
#To see how many rows and columns present in the dataset

haberman.shape

#305 rows and 4 columns present in the dataset
haberman.info()
#checking for any missing values

haberman.isnull().sum()
#checking how many values does class has

haberman["survival_status"].value_counts()
#Seperating the survived and not_Survived data

survived_patients = haberman[haberman['survival_status'] == 1]

not_survived_patients = haberman[haberman['survival_status'] == 2]
#Checking whether data is splitted properly or not

survived_patients.head()
not_survived_patients.head()
plt.figure(2,figsize=(14,4))

plt.subplot(131)

plt.plot(survived_patients['age'],np.zeros_like(survived_patients['age']),'o',label='survived')

plt.plot(not_survived_patients['age'],np.zeros_like(not_survived_patients['age']),'o',label='not-survived')

plt.legend()

plt.xlabel('age')

plt.title('Survival_Status Based on Age')



plt.subplot(132)

plt.plot(survived_patients['aux_nodes_detected'],np.zeros_like(survived_patients['aux_nodes_detected']),'o',label='survived')

plt.plot(not_survived_patients['aux_nodes_detected'],np.zeros_like(not_survived_patients['aux_nodes_detected']),'o',label='not-survived')

plt.legend()

plt.xlabel('aux_nodes_detected')

plt.title('Survival Status Bases on aux_nodes_detected')



plt.subplot(133)

plt.plot(survived_patients['year_of_operation'],np.zeros_like(survived_patients['year_of_operation']),'o',label='survived')

plt.plot(not_survived_patients['year_of_operation'],np.zeros_like(not_survived_patients['year_of_operation']),'o',label='not-survived')

plt.legend()

plt.xlabel('year_of_operation')

plt.title('Survival Status Bases on year_of_operation')
sns.FacetGrid(haberman, hue="survival_status", size=5).map(sns.distplot, "age").add_legend()

plt.title('Histogram for survival_status based on age')

plt.show()
sns.FacetGrid(haberman, hue="survival_status", size=5).map(sns.distplot, "year_of_operation").add_legend()

plt.title('Histogram for survival_status based on year_of_operation')

plt.show()
sns.FacetGrid(haberman, hue="survival_status", size=5).map(sns.distplot, "aux_nodes_detected").add_legend()

plt.title('Histogram for survival_status based on auxillary_nodes_detected')

plt.show()
plt.figure(3,figsize=(20,5))

for idx, feature in enumerate(list(survived_patients.columns)[:-1]):

    plt.subplot(1, 3, idx+1)

    

    print("="*30+"SURVIVED_PATIENT"+"="*30)

    print("********* "+feature+" *********")

    counts, bin_edges = np.histogram(survived_patients[feature], bins=10, density=True)

    print("Bin Edges: {}".format(bin_edges))

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, label = 'pdf_survived')

    plt.plot(bin_edges[1:], cdf, label= 'cdf_survived')

    

    print("="*30+"NOT_SURVIVED_PATIENT"+"="*30)

    counts, bin_edges = np.histogram(not_survived_patients[feature], bins=10, density=True)

    print("Bin Edges: {}".format(bin_edges))

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, label = 'pdf_not_survived')

    plt.plot(bin_edges[1:], cdf, label= 'cdf_not_survived')

    

    plt.title('pdf & cdf for patients based on '+feature)

    plt.legend()

    plt.xlabel(feature)
sns.boxplot(x='survival_status',y='age', data=haberman)

plt.title('box_plot based on age')

plt.show()
sns.boxplot(x='survival_status',y='year_of_operation', data=haberman)

plt.title('box_plot based on year_of_operation')
sns.boxplot(x='survival_status',y='aux_nodes_detected', data=haberman)

plt.title('box_plot based on auxillary_nodes_detected')
sns.violinplot(x="survival_status", y="age", data=haberman, size=8)

plt.title('violin_plot based on age')
sns.violinplot(x="survival_status", y="year_of_operation", data=haberman, size=8)

plt.title('violin_plot based on year_of_operation')
sns.violinplot(x="survival_status", y="aux_nodes_detected", data=haberman, size=8)

plt.title('violin_plot based on auxillary_nodes_detected')
sns.set_style("whitegrid")

sns.pairplot(haberman, hue="survival_status",vars=['age','year_of_operation','aux_nodes_detected'], size=4)

plt.show()
sns.jointplot(x="age", y="year_of_operation", data=haberman, kind="kde")

plt.title('Contour Plot age vs year_of_operation')

plt.show()
sns.jointplot(y="aux_nodes_detected", x="age", data=haberman, kind="kde")

plt.title('Contour Plot age vs auxillary_nodes_detected')

plt.show()
sns.jointplot(x="year_of_operation", y="aux_nodes_detected", data=haberman, kind="kde");

plt.title('Contour Plot year_of_operation vs auxillary_nodes_detected')

plt.show()