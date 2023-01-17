import numpy as np               # Linear Algebra Library for Scientific Computing
import pandas as pd              # Pandas - Python Data Analysis Library
import matplotlib.pyplot as plt  # Plotting Library
import seaborn as sns            # Python Visualization Library based on matplotlib

%matplotlib inline             

dataset = pd.read_csv('../input/haberman.csv',names=['Age','Operation_Year','Axil_nodes','Survival_Status'])
# Checking First 10 entries of Dataset
dataset.head(10)
dataset['Survival_Status'] = dataset['Survival_Status'].map(lambda a:'True' if a==1 else 'False')
# Again Checking top 10 elements in  Dataset
dataset.head(10)
# Checking Column names of Dataset of the Dataset
print(dataset.columns)
# Getting Description of our Dataset

dataset.describe()
# Checking Shape of the Dataset :- (nrows,ncols)
dataset.shape
# Class Labels
dataset['Survival_Status'].unique()
# 2 Class Labels
# No of Data points belonging to each Class Label
dataset['Survival_Status'].value_counts()
# Splitting the Data Based Of their Class Label
survived = dataset.loc[dataset['Survival_Status'] == 'True']
not_survived = dataset.loc[dataset['Survival_Status'] == 'False']
# Plotting Distribution Plot based on each Feature of the Patient

#Distribution Plot for Age Feature
sns.set_style("whitegrid")
sns.FacetGrid(dataset,hue='Survival_Status',size=5)\
.map(sns.distplot,'Age').add_legend()
plt.title("DistPlot for 'Age'")
#Distribution Plot for Operation_year Feature
sns.set_style("whitegrid")
sns.FacetGrid(dataset,hue='Survival_Status',size=5)\
.map(sns.distplot,'Operation_Year').add_legend()
plt.title("DistPlot for 'Operation Year'")
#Distribution Plot for Axil_nodes Feature
sns.set_style("whitegrid")
sns.FacetGrid(dataset,hue='Survival_Status',size=5)\
.map(sns.distplot,'Axil_nodes').add_legend()
plt.title("DistPlot for 'Axil_nodes'")
# Barplot (Aggregates the Categorical Data(Survival_Status) based of some Function(By default - mean()))
sns.set_style("whitegrid")
sns.barplot(x='Survival_Status',y='Axil_nodes',data=dataset)
plt.title("Survival_Status vs Axil_nodes")
sns.set_style("whitegrid")
sns.barplot(x='Survival_Status',y='Age',data=dataset)
plt.title("Survival_Status vs Age")
sns.set_style("whitegrid")
sns.barplot(x='Survival_Status',y='Operation_Year',data=dataset)
plt.title("Survival_Status vs Operation Year")
#  Lets Check the description of Survived and not_survived Patients

#CHECKING DESCRIPTION OF OUR SURVIVED PATIENTS
survived.describe()
#CHECKING DESCRIPTION OF OUR NOT_SURVIVED PATIENTS
not_survived.describe()
# Plotting PDF(Probability Density Function) and CDF(Cumulative Distribution Function)
sns.set_style("whitegrid")
#Survived
count , bin_edge = np.histogram(survived['Axil_nodes'],bins=10,density=True)
pdf = count/sum(count)
cdf= np.cumsum(pdf)
plt.plot(bin_edge[1:],pdf)
plt.plot(bin_edge[1:],cdf)

#not_Survived
count , bin_edge = np.histogram(not_survived['Axil_nodes'],bins=10,density=True)
pdf = count/sum(count)
cdf= np.cumsum(pdf)
plt.plot(bin_edge[1:],pdf)
plt.plot(bin_edge[1:],cdf)
plt.legend(['1:PDF_Survived' , '1:CDF_Survived','2:PDF_notSurvived' , '2:CDF_notSurvived'],loc='center right')
plt.xlabel("Axil nodes Survival Prob-->")
plt.title("CDF - Axil nodes")
plt.tight_layout()
#MEDIANS

print("Medians ")
print("Survived --> ",np.median(survived['Axil_nodes']))
print("Not Survived --> ",np.median(not_survived['Axil_nodes']))

print("\nMEdian Absolute Deviation")
from statsmodels import robust
print("Survived --> ",robust.mad(survived['Axil_nodes']))
print("Not Survived --> ",robust.mad(not_survived['Axil_nodes']))
# Boxplot
sns.boxplot(x = 'Survival_Status',y = 'Axil_nodes',data =dataset)
plt.title("Survival Status vs Axil_nodes")
plt.tight_layout()
# Violin Plot
sns.violinplot(x = 'Survival_Status',y = 'Axil_nodes',data =dataset,hue='Survival_Status')
plt.title("Survival Status vs Axil_nodes")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# PAIRPLOT
sns.pairplot(data = dataset,hue = 'Survival_Status',kind ='scatter',size=3)
plt.suptitle("Pair Plot between Features of Haberman Dataset")
plt.tight_layout()

# Correlation Using Heatmap
sns.heatmap(dataset.corr(),cmap='coolwarm',annot=True)
plt.title("Correlation between features of Dataset")
plt.tight_layout()