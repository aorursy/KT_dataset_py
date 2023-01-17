import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
data.info()
missing = data[data.Albumin_and_Globulin_Ratio.isna()]

for index, row in missing.iterrows():
    age, disease, gender = row["Age"], row["Dataset"], row["Gender"]
    new_table = data[(data["Age"] == age) & (data["Gender"] == gender) & (data["Dataset"] == disease)]
    print(age, disease, gender, new_table["Albumin_and_Globulin_Ratio"].mean())
    data.set_value(index, "Albumin_and_Globulin_Ratio", new_table["Albumin_and_Globulin_Ratio"].mean())
data.info()
data.columns
data = data.rename(columns = {"Alkaline_Phosphotase": "ALP", "Alamine_Aminotransferase": "ALT", "Aspartate_Aminotransferase": "AST", "Total_Protiens":"Protein", "Albumin_and_Globulin_Ratio": "AGR", "Dataset": "Liver Patient"})

data["Liver Patient"].replace(2, 0, inplace = True)
f, axes = plt.subplots(1, 2, figsize = (10, 5))
f.tight_layout(pad = 5)
sns.countplot(data = data, x = "Gender", ax = axes[0])
sns.countplot(data = data, x = "Liver Patient", ax = axes[1])
pd.pivot_table(data, index = "Liver Patient")
pd.pivot_table(data[[col for col in data.columns if col != "Liver Patient"]], index = "Gender")
pd.pivot_table(data, index = ["Liver Patient", "Gender"])
data.describe()
outliers = data[["Total_Bilirubin", "Direct_Bilirubin", "ALP", "ALT", "AST", "Protein", "Albumin", "AGR"]].copy()
index = outliers[(np.abs(stats.zscore(outliers)) < 2.5).all(axis = 1)].index
index2 = outliers[(np.abs(stats.zscore(outliers)) < 2).all(axis = 1)].index

outliers_25z = data.iloc[index, ].copy()
outliers_2z = data.iloc[index2, ].copy()
outliers_25z.describe()
outliers_2z.describe()
sns.set_style("darkgrid")
graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "AST")
graph
graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALT")
graph
graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALP")
graph
graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "AST")
graph
graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALT")
graph
graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALP")
graph
f, axes = plt.subplots(1, 3, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "AST", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "ALT", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "ALP", hue = "Gender", data = data, orient = 'v', ax = axes[2], showfliers = False)
f, axes = plt.subplots(1, 2, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "Total_Bilirubin", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "Direct_Bilirubin", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)
f, axes = plt.subplots(1, 3, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "Protein", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "Albumin", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "AGR", hue = "Gender", data = data, orient = 'v', ax = axes[2], showfliers = False)