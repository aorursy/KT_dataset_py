import pandas as pd

from sklearn.preprocessing import StandardScaler
cancer = pd.read_csv("../input/data.csv")
cancer.head()
cancer.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
X = cancer.drop('diagnosis', axis=1)

Y = cancer['diagnosis']
sc = StandardScaler()

std_X = sc.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA()

pca.fit(std_X)

v = pca.explained_variance_
tot = v.sum()

sum = 0

for i in range(v.shape[0]):

    sum = sum + v[i]/tot

    if sum>0.99:

        break

i
pca.n_components = 16

std_x_pca = pca.transform(std_X)

std_x_pca