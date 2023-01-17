import pandas as pd

from sklearn import preprocessing

from matplotlib import pyplot as plt

from sklearn import covariance

import seaborn as sns

import numpy as np

from numpy import linalg as LA

from sklearn.decomposition import PCA
Input_Data = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")
Input_Data.describe()
Input_Data.head()
DataColumnNames = Input_Data.columns

Input_Data_List_Temp = []

for DataColumnName in DataColumnNames:

    Input_Data_List_Temp.append(preprocessing.normalize([Input_Data[DataColumnName].to_list()],norm='max')[0])

Input_Data_List = np.transpose(Input_Data_List_Temp)
plt.figure(figsize=(10,10))

sns.set(font_scale=1.5)

hm = sns.heatmap(pd.DataFrame(Input_Data_List).corr(),annot=True,annot_kws={"size":8},xticklabels=DataColumnNames, yticklabels=DataColumnNames)

plt.title('Covariance matrix showing correlation coefficients')

plt.tight_layout()

plt.show()
Flavanoids = Input_Data['Flavanoids'].to_list()

Total_Phenols = Input_Data['Total_Phenols'].to_list()

Flavanoids_Normalized = preprocessing.normalize([Flavanoids],norm='max')[0]

Total_Phenols_Normalized = preprocessing.normalize([Total_Phenols],norm='max')[0]

FlavanoidsMean = np.array(Flavanoids_Normalized).mean()

Total_PhenolsMean = np.array(Total_Phenols_Normalized).mean()
plt.scatter(Flavanoids_Normalized,Total_Phenols_Normalized,marker='x')
SelectedFeaturesTransposed = []

SelectedFeaturesTransposed.append(Flavanoids_Normalized)

SelectedFeaturesTransposed.append(Total_Phenols_Normalized)

SelectedFeatures = np.transpose(SelectedFeaturesTransposed)
FlavanoidsSubtracted = np.add(SelectedFeaturesTransposed[0],-FlavanoidsMean)

Total_PhenolsSubtracted = np.add(SelectedFeaturesTransposed[1],-Total_PhenolsMean)
plt.scatter(FlavanoidsSubtracted,Total_PhenolsSubtracted,marker='x')
SelectedFeaturesSubtracted = np.array([FlavanoidsSubtracted,Total_PhenolsSubtracted])
S = SelectedFeaturesSubtracted.dot(SelectedFeaturesSubtracted.T) / 178
S
EigenValue,EigenVector = LA.eig(S)

print(EigenVector)

print(EigenValue)
PC1 = EigenVector.T[0].dot(SelectedFeaturesTransposed)

print(np.var(PC1))
PC2 = EigenVector.T[1].dot(SelectedFeaturesTransposed)

print(np.var(PC2))
plt.scatter(FlavanoidsSubtracted,Total_PhenolsSubtracted,marker='x')

plt.quiver([0, 0], [0, 0], EigenVector[0], EigenVector[1], scale=3, color=['r','g'], label=['PC1','PC2'])

plt.text(0.2,0.3,'PC1')

plt.text(-0.2,0.3,'PC2')
plt.scatter(PC1,PC2,marker='x')

plt.title("PCA Plot")

plt.xlabel("PC1")

plt.ylabel("PC2")
# Provide the number of principal components required - 2 for plotting points on a plane

pca = PCA(n_components=2)
pca.fit(SelectedFeatures)

# print the Principal component vectors - Eigen vectors

print(pca.components_)

# Print the covariance matrix

print(pca.get_covariance())
PC1 = pca.components_[0].dot(SelectedFeaturesTransposed)

print(np.var(PC1))

PC2 = pca.components_[1].dot(SelectedFeaturesTransposed)

print(np.var(PC2))
plt.scatter(PC1,PC2,marker='x')

plt.title("PCA Plot")

plt.xlabel("PC1")

plt.ylabel("PC2")