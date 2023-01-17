import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
path_cancer_data = "../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv"

cancer_data = pd.read_csv(path_cancer_data)
print(cancer_data.keys())
plt.figure(figsize=(6,4))

sns.lineplot(x=cancer_data["radius_mean"],y=cancer_data["perimeter_mean"], hue=cancer_data["diagnosis"])
sns.barplot(x=cancer_data['diagnosis'], y=cancer_data['area_mean'])
sns.barplot(x=cancer_data[:10].index, y=cancer_data['area_mean'][:10], hue=cancer_data['diagnosis'][:10])
labels = ['smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

heat_cancer_data = cancer_data[labels]

sns.heatmap(data=heat_cancer_data[:10], annot=True)
plt.figure(figsize=(6,4))

sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_mean"], hue=cancer_data["diagnosis"])
sns.regplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_mean"]) 
sns.lmplot(x='area_mean', y='smoothness_mean', hue='diagnosis' , data=cancer_data)
sns.swarmplot(x=cancer_data['diagnosis'], y=cancer_data['smoothness_mean'])
sns.distplot(a=cancer_data['perimeter_mean'], kde=False)
sns.kdeplot(data=cancer_data['perimeter_mean'], shade=True)
sns.jointplot(x=cancer_data['perimeter_mean'], y=cancer_data['smoothness_mean'], kind="kde")
sns.set_style("dark")