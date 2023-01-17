import pandas as pd
import numpy as np
wine = pd.read_csv("../input/wine-dataset/wine.csv")
wine.head()
wine.shape
wine.describe()
from matplotlib import pyplot as plt
wine['Class'].value_counts().plot.bar(title="Distribution of Wine Classes")
#remove the class column from wine, and save as label
label = wine.pop('Class')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(wine)
#the results will show that PC1 dominates for the original data
print("variance explained", pca.explained_variance_ratio_, "singular_values", pca.singular_values_)
#transform the data according to the PCA results
wine_transformed = pca.transform(wine)
#wine_transformed = pd.DataFrame.from_records(pca.transform(wine))
#wine_transformed['label'] = label
#apply standardscaler to scale the data
from sklearn.preprocessing import StandardScaler
wine_scaled = StandardScaler().fit_transform(wine)
first_column = np.array(wine_scaled[:, 0])
print(first_column[:10], "...")
print("mean %.2f" % first_column.mean())
print("var %.2f" % first_column.var())
#apply PCA to scaled data
pca = PCA(n_components=2)
pca.fit(wine_scaled)
print("variance explained", pca.explained_variance_ratio_, "singular_values", pca.singular_values_)
print(pca)
wine_scaled_transformed = pca.transform(wine_scaled)
# comparison of scatter plots of standarized and original dataset after dimension reduction
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(wine_transformed[label == l, 0],
                wine_transformed[label == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )
for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(wine_scaled_transformed[label == l, 0],
                wine_scaled_transformed[label == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )
ax1.set_title('original wine dataset')
ax2.set_title('standardized wine dataset')

for ax in (ax1, ax2):
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()
plt.show()