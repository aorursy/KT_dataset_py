import pandas as pd

# load dataset into Pandas DataFrame

df = pd.read_csv("../input/iris.csv")

df.info()
from sklearn.preprocessing import StandardScaler

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Separating out the features

x = df.loc[:, features].values

# Separating out the target

y = df.loc[:,['species']].values

# Standardizing the features

x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])





finalDf = pd.concat([principalDf, df[['species']]], axis = 1)



finalDf.head()

#1st and 2nd PCs decomposition axis

print(pca.components_)

#Variance explained by 1st and 2nd PCs

pca.explained_variance_ratio_

# total variance explained by two PCs = 0.9580
pca.get_precision()