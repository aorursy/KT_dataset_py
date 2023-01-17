import numpy as np , pandas as pd 
import seaborn as snd 
import matplotlib.pyplot as plt
customers = pd.read_csv('../input/patient/Mall_Patient.csv')
customers.head(10)
# Check to see descriptive statistics
customers.describe()
# See the distribution of gender to recognize different distributions
import seaborn as sns
sns.countplot(x='Gender', data=customers);
plt.title('Distribution of Gender');
# Histogram of ages
customers.hist('Age', bins=35);
plt.title('Distribution of Age');
plt.xlabel('Age');
# Histogram of ages by gender
plt.hist('Age', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Age', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Age by Gender');
plt.xlabel('Age');
plt.legend();

# Histogram of income
customers.hist('Slice position in .nii.gz');
plt.title('Slice position  Distribution in SIRM Case');
plt.xlabel('SIRM Case');

# Histogram of income by gender
plt.hist('Slice position in .nii.gz', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Slice position in .nii.gz', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Slice position by Gender');
plt.xlabel('Slice Case');
plt.legend();

# Create data sets by gender
male_customers = customers[customers['Gender'] == 'Male']
female_customers = customers[customers['Gender'] == 'Female']
# Print the average SIRM Case for men and women
print(male_customers['SIRM Case'].mean())
print(female_customers['SIRM Case'].mean())
sns.scatterplot('Age', 'Slice position in .nii.gz', hue='Gender', data=customers);
plt.title('Age to Slice position, Colored by Gender');

sns.scatterplot('Age', 'SIRM Case', hue='Gender', data=customers);
plt.title('Age to SIRM Case, Colored by Gender');

sns.heatmap(female_customers.corr(), annot=True);
plt.title('Correlation Heatmap - Female');

sns.heatmap(male_customers.corr(), annot=True);
plt.title('Correlation Heatmap - Male');
sns.lmplot('Age','Slice position in .nii.gz', data=male_customers);
plt.title('Age to Slice position , Male Only');

sns.scatterplot('Slice position in .nii.gz', 'SIRM Case', hue='Gender', data=customers);
plt.title('Slice position to SIRM Case, Colored by Gender');