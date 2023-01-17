# Importing the Important Libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# Importing the dataset

try :
    columnNamesList = ["PatientAge", "PatientYearOperation", "NumberAxillaryNodes", "SurvivalStatus"]
    cancer_data = pd.read_csv('./../input/haberman.csv', names=columnNamesList);
    print("Loaded the dataset successfully.")
except(FileNotFoundError):
    print("Make sure you check the path of the file")
except:
    print("An Error occured.")

# Printing the top 5 rows for inital analysis of the dataset

cancer_data.head()
unique_valuesList = list(cancer_data["SurvivalStatus"].unique())
print(cancer_data.info())
print("\nThe Unique values in the class attribute column are : {}".format(unique_valuesList))
# Data Preprocessing
# Since the class attributes of the dataset i.e Survival Status is numeric, it needs to properly mapped.
# 1 -> 'Survived' and 2 -> 'Died'

cancer_data["SurvivalStatus"].replace(to_replace = {
    1: "Survived",
    2: "Died"
}, inplace = True)

print(cancer_data.info())
print(cancer_data.describe())
print(cancer_data["SurvivalStatus"].value_counts())
print("\n" + str(cancer_data["SurvivalStatus"].value_counts(normalize = True)))
print(cancer_data.isnull().any())
print("--------------------------")
print(cancer_data.count())
plt.close()
cancer_data.hist()
plt.figure(figsize=(18,8))
plt.subplot(1, 3, 2)
plt.title("PDF - CDF Plot")

for index, feature in enumerate(list(cancer_data.columns)[:-1]):
    plt.subplot(1, 3, index+1)
    counts, bin_edges = np.histogram(cancer_data[feature], bins = 10, density = True)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(feature)
    plt.grid()

plt.show()

plt.close()
plt.figure(figsize=(18,8))
for index, columnY in enumerate(list(cancer_data.columns)[:-1]):
    plt.subplot(1, 3, index+1)
    sns.boxplot(x="SurvivalStatus", y=columnY, data=cancer_data, palette='husl')
    plt.grid()
plt.subplot(1, 3, 2)
plt.title("Box-Plot / Whiskers Plot")
plt.show()
plt.close()
plt.figure(figsize=(18,8))
for index, columnY in enumerate(list(cancer_data.columns)[:-1]):
    plt.subplot(1, 3, index+1)
    sns.violinplot(x="SurvivalStatus", y=columnY, data=cancer_data)
    plt.grid()
plt.show()

a = sns.FacetGrid(data=cancer_data, hue="SurvivalStatus", height=4)\
    .map(plt.scatter, "PatientAge", "NumberAxillaryNodes")\
    .add_legend()

b = sns.FacetGrid(data=cancer_data, hue="SurvivalStatus", height=4)\
    .map(plt.scatter, "PatientAge", "PatientYearOperation")\
    .add_legend()

c = sns.FacetGrid(data=cancer_data, hue="SurvivalStatus", height=4)\
    .map(plt.scatter, "NumberAxillaryNodes", "PatientYearOperation")\
    .add_legend()
plt.show()
plt.close()
sns.pairplot(data=cancer_data, hue="SurvivalStatus", height=5)
plt.grid()
plt.show()