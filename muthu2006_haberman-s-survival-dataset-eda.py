#Importing required modules and dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



# pd.set_option('display.max_rows', 500)

# pd.set_option('display.max_columns', 500)

print(os.listdir("../input"))

data = pd.read_csv('../input/haberman.csv', header = None, names = ['age_of_patient', 'year_of_surgery', 'positive_axillary_nodes', 'survival_status'])

print("Modules and dataset imported succesfully")
data.head()
print("\nThere are {} patients in the list".format(len(data.index)));

print("\nColumn names are {}".format(data.columns.tolist()))

# Converting survival_status column values to categorical: 1 - Survived, 2 - Died

data.survival_status.replace(to_replace=[1, 2], value=['Survived', 'Died'], inplace=True)

print("\nFull summary of data as follows:\n")

data.info()

print("\nMissing data% stats as follows:\n\n{}".format((data.isnull().sum() / len(data)) * 100))
# Now the dataframe looks like

data.head()
data.describe()
data.survival_status.value_counts()  # Gives the count of Patients survived and died
# Countplot of Patients Survived vs Patients Died

sns.set_style('whitegrid')

fig1, ax = plt.subplots(1, 2, figsize=(15,7))

fig1 = sns.countplot(x=data.survival_status, hue='survival_status', data=data, ax=ax[0])

fig1.set_xlabel('Survival Status of Patients', fontsize=15)

fig1.set_ylabel('Number of Patients', fontsize=15)

fig1.set_title('Number of Patients Survived vs Patients Died', fontsize=15)

fig1.legend(title='Survival Status')



# Pieplot showing percentages of Patients Survived and Patients Died

ax[1].pie(x=data.survival_status.value_counts(), labels=['Survived', 'Died'], autopct='%.2f%%')

ax[1].axis('equal')

ax[1].set_title('Percentage of Patients Survived vs Patients Died', fontsize=15)

ax[1].legend(title='Survial Status')

plt.show()
# Countplot of Age vs Number of Patients Relationship

sns.set_style('white')

fig2, ax = plt.subplots(4, 1, figsize=(20,25))

fig2 = sns.countplot(data.age_of_patient, ax=ax[0])

fig2.set_xlabel('Age of patients', fontsize=15)

fig2.set_ylabel('Number of patients', fontsize=15)

fig2.set_title('Age vs Number of Patients', fontsize=15)



# Histogram of Age group vs Number of Patients Relationship

ax[1].hist(x=data.age_of_patient, bins=[30,35,40,45,50,55,60,65,70,75,80,85])

ax[1].set_xlabel('Age of patients', fontsize=15)

ax[1].set_ylabel('Number of patients', fontsize=15)

ax[1].set_title('Age vs Number of Patients', fontsize=15)



# Distplot of distribution of patients age

sns.distplot(data.age_of_patient, ax=ax[2])

ax[2].set_xlabel('Age of patients', fontsize=15)

ax[2].set_ylabel('Density', fontsize=15)

ax[2].set_title('Density distribution of Patients Age', fontsize=15)



#PDF and CDF of Patient age

counts, bin_edges = np.histogram(data.age_of_patient)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label='PDF')

plt.plot(bin_edges[1:], cdf, label='CDF')

plt.xlabel('Age of patient', fontsize=15)

plt.title('PDF and CDF of Patient age', fontsize=15)

plt.legend(fontsize=15)

plt.show()
data.age_of_patient.value_counts().sort_index()  # Gives the count of patients per age
# Countplot of Age vs Survival Relationship

sns.set_style('whitegrid')

fig1, ax = plt.subplots(1, 1, figsize=(15,7))

fig1 = sns.countplot(x=data.age_of_patient, hue='survival_status', data=data, ax=ax)

fig1.set_xlabel('Age of patient', fontsize=15)

fig1.set_ylabel('Number of patients Survived/Died', fontsize=15)

fig1.set_title('Age vs Survival Relationship', fontsize=15)

plt.legend(bbox_to_anchor=(1,1), loc='upper right', title='Survival Status')

plt.show()
# Swarmplot of Age vs Survival Relationship

sns.set_style('white')

fig, ax = plt.subplots(figsize=(7, 5))

sns.swarmplot(data=data, y='survival_status', x='age_of_patient', ax=ax)

# sns.swarmplot(data=data, y='survivalStatus', x='ageOfPatient', hue='survivalStatus', ax=ax)

ax.set_xlabel('Age of patient', fontsize=15)

ax.set_ylabel('Survival status', fontsize=15)

ax.set_title('Age vs Survival Relationship', fontsize=15)

# ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  # a legend would be too obvious here, hence commented

plt.show()
pd.crosstab(data.age_of_patient, data.survival_status) # Gives the number of Patients Survived/Died per age
# # data.groupby('ageOfPatient')['survivalStatus'].value_counts()

# Following code gives Patient survival % per age.

group = data.groupby('age_of_patient')['survival_status']

for name, group in group:

    try:

        print("Age of patient : "+str(name)+", Patient Survival rate is : "+

              str(round((group.value_counts()['Survived']/group.value_counts().sum())*100,2))+"%")

    except:

        print("Age of patient : "+str(name)+", Patient Survival rate is : 0%")
# Box plot to give 5 number summary

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(10, 7))

sns.boxplot(data=data, y='survival_status', x='age_of_patient', hue='survival_status', ax=ax)

ax.set_xlabel('Age of patient', fontsize=15)

ax.set_ylabel('Survival status', fontsize=15)

ax.set_title('Age vs Survival Relationship', fontsize=15)

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
# Violin plot

sns.set_style('white')

fig, ax = plt.subplots(figsize=(10, 7))

sns.violinplot(data=data, x='survival_status', y='age_of_patient', hue='survival_status', ax=ax)

ax.set_xlabel('Survival status', fontsize=15)

ax.set_ylabel('Age of patient', fontsize=15)

ax.set_title('Age vs Survival Relationship', fontsize=15)

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
# To calculate Q1, Median(Q2), Q3, IQR, Min, Max etc

group = data.groupby('survival_status')['age_of_patient']

for name, group in group:

    print("\nPatient status : "+str(name))

    print("Lowest age is "+str(np.min(group)))

    print("Highest age is "+str(np.max(group)))

    print("Median age is "+str(np.median(group)))

    print("1st Quartile (Q1) is "+str(np.percentile(group, 25)))

    print("2nd Quartile (Q2) is "+str(np.percentile(group, 50))+ " which should be same as Median")

    print("3rd Quartile (Q3) is "+str(np.percentile(group, 75)))

    print("IQR is "+str(abs((np.percentile(group, 25))-(np.percentile(group, 75)))))
# Distplot of distribution of Patients age

ax = sns.FacetGrid(data, hue='survival_status', height=5)

ax.map(sns.distplot, 'age_of_patient')

ax.set_xlabels('Age of patients', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

plt.show()
# Distplot of distribution of Patients age

ax = sns.FacetGrid(data, col='survival_status', hue='survival_status', height=5)

ax.map(sns.distplot, 'age_of_patient')

ax.set_xlabels('Age of patients', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

plt.show()
# Countplot of Year of surgery vs Number of Patients undergone surgery

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(10,5))

fig = sns.countplot(data.year_of_surgery, ax=ax)

fig.set_xlabel('Year of surgery', fontsize=15)

fig.set_ylabel('Number of Patients undergone surgery', fontsize=15)

fig.set_title('Year of surgery vs Number of Patients undergone surgery', fontsize=15)

fig.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

plt.show()
data.year_of_surgery.value_counts()  # Gives number of Patients undergone surgery per year
# Countplot of Year of surgery vs Survival Relationship

sns.set_style('whitegrid')

fig1, ax = plt.subplots(1, 1, figsize=(15,7))

fig1 = sns.countplot(x=data.year_of_surgery, hue='survival_status', data=data, ax=ax)

fig1.set_xlabel('Year of surgery', fontsize=15)

fig1.set_ylabel('Number of Patients Survived/Died', fontsize=15)

fig1.set_title('Year of surgery vs Survival Relationship', fontsize=15)

fig1.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

fig1.legend(title='Survival Status')

plt.show()
pd.crosstab(data.year_of_surgery, data.survival_status) # Gives the number of Patients Survived/Died per year
# data.groupby('yearOfSurgery')['survivalStatus'].value_counts()

group = data.groupby('year_of_surgery')['survival_status']

for name, group in group:

    print("Year of surgery : "+"19"+str(name)+", Patient Survival rate is : "+

          str(round((group.value_counts()['Survived']/group.value_counts().sum())*100,2))+"%")
# Distplot of distribution of surgery year

ax = sns.FacetGrid(data, hue='survival_status', height=5)

ax.map(sns.distplot, 'year_of_surgery')

ax.set_xlabels('Year of surgery', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

#ax.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

plt.show()
# Distplot of distribution of surgery year

ax = sns.FacetGrid(data, col='survival_status', hue='survival_status', height=5)

ax.map(sns.distplot, 'year_of_surgery')

ax.set_xlabels('Year of surgery', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

#ax.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

plt.show()
# Countplot of Positive Axillary nodes detected vs Number of Patients affected

sns.set_style('whitegrid')

fig3, ax = plt.subplots(figsize=(20,5))

fig3 = sns.countplot(data.positive_axillary_nodes, ax=ax)

fig3.set_xlabel('Positive Axillary Nodes detected', fontsize=15)

fig3.set_ylabel('Number of Patients affected', fontsize=15)

fig3.set_title('Positive Axillary Nodes detected', fontsize=15)

plt.show()
data.positive_axillary_nodes.value_counts()  # Gives count of positive Axillary nodes and corresponding number of patients
# Countplot of Positive Axillary nodes detected vs Survival Relationship

sns.set_style('whitegrid')

fig1, ax = plt.subplots(1, 1, figsize=(15,7))

fig1 = sns.countplot(x=data.positive_axillary_nodes, hue='survival_status', data=data, ax=ax)

fig1.set_xlabel('Number of Positive Axillary Nodes detected', fontsize=15)

fig1.set_ylabel('Number of Patients Survived/Died', fontsize=15)

fig1.set_title('Positive Axillary Nodes detected vs Survival Relationship', fontsize=15)

plt.legend(bbox_to_anchor=(1,1), loc='upper right', title='Survival Status')

plt.show()
# Swarmplot of Positive Axillary nodes detected vs Survival Relationship

sns.set_style('white')

fig, ax = plt.subplots()

sns.swarmplot(data=data, y='survival_status', x='positive_axillary_nodes', ax=ax)

ax.set_xlabel('Positive Axillary nodes detected', fontsize=15)

ax.set_ylabel('Survival status', fontsize=15)

ax.set_title('Positive Axillary nodes detected vs Survival Relationship', fontsize=15)

plt.show()
pd.crosstab(data.positive_axillary_nodes, data.survival_status)  # Gives the number of Patients Survived/Died per Positive Axillary nodes detected
# data.positiveAxNodes.value_counts()

# Following code gives Patient survival % per number of nodes detected.

group = data.groupby('positive_axillary_nodes')['survival_status']

for name, group in group:

    try:

        print("Axillary Nodes detected : "+str(name)+", Patient Survival rate is : "+

              str(round((group.value_counts()['Survived']/group.value_counts().sum())*100,2))+"%")

    except:

        print("Axillary Nodes detected : "+str(name)+", Patient Survival rate is : 0%")
# Box plot to give 5 number summary

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(10, 7))

sns.boxplot(data=data, y='survival_status', x='positive_axillary_nodes', hue='survival_status', ax=ax)

ax.set_xlabel('Positive Axillary nodes detected', fontsize=15)

ax.set_ylabel('Survival status', fontsize=15)

ax.set_title('Positive Axillary nodes detected vs Survival Relationship', fontsize=15)

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
# Violin plot

sns.set_style('white')

fig, ax = plt.subplots(figsize=(10, 7))

sns.violinplot(data=data, x='survival_status', y='positive_axillary_nodes', hue='survival_status', ax=ax)

ax.set_xlabel('Survival status', fontsize=15)

ax.set_ylabel('Positive Axillary nodes detected', fontsize=15)

ax.set_title('Positive Axillary nodes detected vs Survival Relationship', fontsize=15)

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
# To calculate Q1, Median(Q2), Q3, IQR, Min, Max etc

group = data.groupby('survival_status')['positive_axillary_nodes']

for name, group in group:

    print("\nPatient status : "+str(name))

    print("Lowest number of positive Axillary nodes detected is "+str(np.min(group)))

    print("Highest number of positive Axillary nodes detected is "+str(np.max(group)))

    print("Median number of positive Axillary nodes detected is "+str(np.median(group)))

    print("1st Quartile (Q1) is "+str(np.percentile(group, 25)))

    print("2nd Quartile (Q2) is "+str(np.percentile(group, 50))+ " which should be same as Median")

    print("3rd Quartile (Q3) is "+str(np.percentile(group, 75)))

    print("IQR is "+str(abs((np.percentile(group, 25))-(np.percentile(group, 75)))))
# Distplot of distribution of Patients age

ax = sns.FacetGrid(data, hue='survival_status', height=5)

ax.map(sns.distplot, 'positive_axillary_nodes')

ax.set_xlabels('Positive Axillary nodes detected', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

plt.show()
# Distplot of distribution of Patients age

ax = sns.FacetGrid(data, hue='survival_status', col='survival_status', height=5)

ax.map(sns.distplot, 'positive_axillary_nodes')

ax.set_xlabels('Positive Axillary nodes detected', fontsize=15)

ax.set_ylabels('Density', fontsize=15)

ax.add_legend(title='Survival Status', fontsize=12)

plt.show()
# Scatter plot

ax1 = sns.FacetGrid(data, hue='survival_status', col='survival_status', height=5)

ax1.map(plt.scatter, 'age_of_patient', 'positive_axillary_nodes')

ax1.set_xlabels('Age of patient', fontsize=15)

ax1.set_ylabels('Positive Axillary nodes detected', fontsize=15)

ax1.add_legend(title='Survival Status', fontsize=12)



# Another one without using col parameter

ax2 = sns.FacetGrid(data, hue='survival_status', height=5)

ax2.map(plt.scatter, 'age_of_patient', 'positive_axillary_nodes')

ax2.set_xlabels('Age of patient', fontsize=15)

ax2.set_ylabels('Positive Axillary nodes detected', fontsize=15)

ax2.add_legend(title='Survival Status', fontsize=12)



# Another way

fig, ax3 = plt.subplots()

sns.scatterplot(x='age_of_patient', y='positive_axillary_nodes', hue='survival_status', data=data, ax=ax3) 

ax3.set_xlabel('Age of patient', fontsize=15) 

ax3.set_ylabel('Positive Axillary nodes detected', fontsize=15) 

ax3.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=12)

plt.show()
# Swarmplot of Age of Patient vs Year of surgery Relationship

sns.set_style('white')

fig, ax = plt.subplots(figsize=(10, 7))

sns.swarmplot(data=data, hue='survival_status', x='year_of_surgery', y='age_of_patient', ax=ax)

ax.set_xlabel('Year of surgery', fontsize=15)

ax.set_ylabel('Age of patient', fontsize=15)

ax.set_title('Age of patient vs Year of surgery Relationship', fontsize=15)

ax.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
# Swarmplot of Year of surgery vs Positive Axillary Nodes detected

sns.set_style('white')

fig, ax = plt.subplots(figsize=(10, 7))

sns.swarmplot(data=data, hue='survival_status', x='year_of_surgery', y='positive_axillary_nodes', ax=ax)

ax.set_xlabel('Year of surgery', fontsize=15)

ax.set_ylabel('Positive Axillary nodes detected', fontsize=15)

ax.set_title('Year of surgery vs Positive Axillary nodes detected', fontsize=15)

ax.set_xticklabels([1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

ax.legend(bbox_to_anchor=(1,1), loc='upper left', title='Survival Status')  

plt.show()
ax = sns.pairplot(data, hue='survival_status', height = 4)

ax.add_legend(title='Survival Status')

plt.show()