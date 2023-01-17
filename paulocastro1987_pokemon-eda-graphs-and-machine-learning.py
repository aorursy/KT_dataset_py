import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Pokemon.csv')
df.head()
df['Type 2'].fillna(df['Type 1'],inplace = True)
def sep(x):

    return x.split('Mega ')[0]
df['Name'] = df['Name'].apply(lambda x: sep(x))
for i in range(1,len(df['Name'])):

    if df['Name'][i] == df['Name'][i-1]:

        df['Name'][i] = 'Mega ' + df['Name'][i]
df.head()
df.drop('#',inplace=True,axis=1)
df.set_index('Name',inplace=True)
df.head()
df[df['Total'] == df['Total'].max()]
df[df['HP'] == df['HP'].max()]
df[df['Attack'] == df['Attack'].max()]
df[df['Defense'] == df['Defense'].max()]
df[df['Sp. Atk'] == df['Sp. Atk'].max()]
df[df['Sp. Def'] == df['Sp. Def'].max()]
df[df['Speed'] == df['Speed'].max()]
df.describe()
a = df.groupby('Type 1')['Total'].idxmax()

b = df.groupby('Type 1')['Total'].max()
c = pd.DataFrame({'Name':a,'Total':b})

c
c.sort_values(by=['Total'])
plt.figure(figsize=(11,6))

sns.set()

plt.title('Distribution of pokemon types')

k = sns.countplot(x = 'Type 1',data=df)

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
plt.figure(figsize=(11,6))

sns.distplot(df['Total'],color='r')
sns.set()

plt.figure(figsize=(14,8))

k = sns.boxplot(x='Type 1',y = 'Total', data = df)

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
plt.figure(figsize=(11,6))

sns.set()

plt.title('Count of pokemon by generations')

k = sns.countplot(x = 'Generation',data=df)

k.set_xticklabels(k.get_xticklabels(), rotation=45)

plt.show()
plt.figure(figsize=(8,8))

j = df.iloc[:,2:9]

sns.heatmap(j.corr(),square=True,robust=True,annot=True,cmap='jet')
def legend(x):

    if x == True:

        return 0

    if x == False:

        return 1
df['Legendary'] = df['Legendary'].apply(lambda x: legend(x))
df.head()
X = df.iloc[:,:10].values
y = df.iloc[:,10].values
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
# Normalizing data

scaler = MinMaxScaler(feature_range = (0, 1))

X[:,2:9] = scaler.fit_transform(X[:,2:9])
a = pd.DataFrame(X)
b = pd.get_dummies(a, columns=[0,1,9])

X = b.iloc[:,0:].values
from sklearn.ensemble import GradientBoostingClassifier
# Number of folds

num_folds = 10

seed = 7



# Number of trees

num_trees = 100



# Separating by folds

kfold = KFold(num_folds, True, random_state = seed)



# Creating the model

modelo = GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)



# Cross Validation

resultado = cross_val_score(modelo, X, y, cv = kfold)



# Print 

print("Accuracy: %.3f" % (resultado.mean() * 100))
from sklearn.decomposition import PCA
# Choosing attributes

pca = PCA(n_components = 4)

fit = pca.fit(X)





print("Variance: %s" % fit.explained_variance_ratio_)

print(np.sum(fit.explained_variance_ratio_))

p = []

x = []

for i in range(1,25):

    pca = PCA(n_components = i)

    fit = pca.fit(X)

    x.append(i)

    p.append(np.sum(fit.explained_variance_ratio_))

x_pca = pca.transform(X)
plt.grid(True)

plt.scatter(x,p)

plt.show()
# Folds

num_folds = 10

seed = 7



# Number of trees

num_trees = 100



# Folds in data

kfold = KFold(num_folds, True, random_state = seed)



# model

modelo = GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)



# Cross Validation

resultado = cross_val_score(modelo, x_pca, y, cv = kfold)



# Printing result with PCA

print("Accuracy: %.3f" % (resultado.mean() * 100))
from xgboost import XGBClassifier
#model

modelo = XGBClassifier(n_estimators = num_trees, random_state = seed)



# Cross Validation

resultado = cross_val_score(modelo, X, y, cv = kfold)



# # Printing result

print("Acurácia: %.3f" % (resultado.mean() * 100))
#model

modelo = XGBClassifier(n_estimators = num_trees, random_state = seed)



# Cross Validation

resultado = cross_val_score(modelo, x_pca, y, cv = kfold)



## Printing result with PCA

print("Accuracy: %.3f" % (resultado.mean() * 100))
from sklearn.neural_network import MLPClassifier
# Model

modelo = MLPClassifier(hidden_layer_sizes=500,max_iter=3000,tol=1e-7,solver='adam')



# Cross Validation

resultado = cross_val_score(modelo, X, y, cv = kfold)



# Print result

print("Accuracy: %.3f" % (resultado.mean() * 100))
# model

modelo = MLPClassifier(hidden_layer_sizes=500,max_iter=3000,tol=1e-7,solver='adam')



# Cross Validation

resultado = cross_val_score(modelo, x_pca, y, cv = kfold)



# # Printing result with PCA

print("Accuracy: %.3f" % (resultado.mean() * 100))