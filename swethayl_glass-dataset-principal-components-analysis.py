from pandas import read_csv
data = read_csv('../input/glass.csv')
data
X = data.drop("Type",1)
y = data["Type"]
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
pca.n_components_