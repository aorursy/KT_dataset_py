import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category= RuntimeWarning)

def warn(*args, **kwargs):

    pass

warnings.warn = warn
haberman_database = pd.read_csv("../input/haberman.csv") #haberman is breast cancer
# print(haberman_database.describe)

print(haberman_database.columns)

haberman_database.describe()
print(haberman_database.head()) #database columns name are not meaningfull
breast_cancer = pd.read_csv("../input/haberman.csv",header=None, names=['age', 'year_of_operation', 'aux_nodes_detected', 'survival_status'])



# print(breast_cancer.head())

# breast_cancer.describe()
# modify the "survival_status"column values to be meaningful as well as categorical

breast_cancer['survival_status'] = breast_cancer['survival_status'].map({1:"survived", 2:"not_survived"})

breast_cancer['survival_status'] = breast_cancer['survival_status'].astype('category')

print(breast_cancer.head())
print(breast_cancer.shape)

print(breast_cancer.columns)

print(breast_cancer['survival_status'].unique())

print(breast_cancer['survival_status'].value_counts())
import numpy as np

survived_patients = breast_cancer.loc[breast_cancer['survival_status']=='survived']

not_survived_patients = breast_cancer.loc[breast_cancer['survival_status']=='not_survived']



plt.figure(1,figsize=(14,4))





plt.subplot(121)

plt.plot(survived_patients["age"], np.zeros_like(survived_patients['age']), 'o',label='survived')

plt.plot(not_survived_patients["age"], np.zeros_like(not_survived_patients['age']), 'o',label='not-survived')

plt.legend()

plt.xlabel('age')

plt.title('surviavl_status based on age')



plt.subplot(122)

plt.plot(survived_patients["aux_nodes_detected"], np.zeros_like(survived_patients['aux_nodes_detected']), 'o',label='survived')

plt.plot(not_survived_patients["aux_nodes_detected"], np.zeros_like(not_survived_patients['aux_nodes_detected']), 'o',label='not-survived')

plt.legend()

plt.xlabel('aux_nodes_detected')

plt.title('surviavl_status based on auxillary nodes')





plt.show()
sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "age").add_legend()

plt.title('Histogram for survival_status based on age')

plt.show()
sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "year_of_operation").add_legend()

plt.title('Histogram for survival_status based on year_of_operation')

plt.show()
sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "aux_nodes_detected").add_legend()

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
sns.boxplot(x='survival_status',y='age', data=breast_cancer)

plt.title('box_plot based on age')

plt.show()
sns.boxplot(x='survival_status',y='year_of_operation', data=breast_cancer)

plt.title('box_plot based on year_of_operation')
sns.boxplot(x='survival_status',y='aux_nodes_detected', data=breast_cancer)

plt.title('box_plot based on auxillary_nodes_detected')
sns.violinplot(x="survival_status", y="age", data=breast_cancer, size=8)

plt.title('violin_plot based on age')
sns.violinplot(x="survival_status", y="year_of_operation", data=breast_cancer, size=8)

plt.title('violin_plot based on year_of_operation')
sns.violinplot(x="survival_status", y="aux_nodes_detected", data=breast_cancer, size=8)

plt.title('violin_plot based on auxillary_nodes_detected')
sns.set_style("whitegrid")

sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"age","aux_nodes_detected").add_legend()

plt.title('2D scatter plot for age vs auxillary_nodes_detected')

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"year_of_operation","aux_nodes_detected").add_legend();

plt.title('2D scatter plot for year_of_operation vs auxillary_nodes_detected')

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"year_of_operation","age").add_legend()

plt.title('2D scatter plot for year_of_operation vs age')

plt.show()
plt.close()

sns.set_style("whitegrid")

sns.pairplot(breast_cancer, hue="survival_status",vars=['age','year_of_operation','aux_nodes_detected'], size=4)

plt.show()
sns.jointplot(x="age", y="year_of_operation", data=breast_cancer, kind="kde")

plt.title('Contour Plot age vs year_of_operation')

plt.show()
sns.jointplot(y="aux_nodes_detected", x="age", data=breast_cancer, kind="kde")

plt.title('Contour Plot age vs auxillary_nodes_detected')

plt.show()
sns.jointplot(x="year_of_operation", y="aux_nodes_detected", data=breast_cancer, kind="kde");

plt.title('Contour Plot year_of_operation vs auxillary_nodes_detected')

plt.show();