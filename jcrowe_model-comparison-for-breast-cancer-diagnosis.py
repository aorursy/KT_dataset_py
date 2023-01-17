import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from scipy.stats import randint as sp_randint



%matplotlib inline

import matplotlib.pyplot as plt
df = pd.read_csv("../input/data.csv",header = 0)

df.head()
# Remove unnecessary columns

df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)
# Encode diagnosis as numerical values(B=0, M=1)

le = preprocessing.LabelEncoder()

le.fit(['M', 'B'])



df['diagnosis'] = le.transform(df['diagnosis'])
df.describe()
from sklearn.decomposition import PCA



# observables = df.loc[:,observe]

observables = df.iloc[:,1:]

pca = PCA(n_components=3)

pca.fit(observables)



# Dimension indexing

dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



# Individual PCA Components

components = pd.DataFrame(np.round(pca.components_, 4), columns = observables.keys())

components.index = dimensions



# Explained variance in PCA

ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

variance_ratios.index = dimensions



print(pd.concat([variance_ratios, components], axis = 1))
# Observe correlation to the diagnosis

tst = df.corr()['diagnosis'].copy()

tst = tst.drop('diagnosis')

tst.sort_values(inplace=True)

tst.plot(kind='bar', alpha=0.6)
# Separate out malignant and benign data for graphing

malignant = df[df['diagnosis'] ==1]

benign = df[df['diagnosis'] ==0]
# Column names to observe in following graphs - mean values only

observe = list(df.columns[1:11]) + ['area_worst'] + ['perimeter_worst']

observables = df.loc[:,observe]
plt.rcParams.update({'font.size': 8})

plot, graphs = plt.subplots(nrows=6, ncols=2, figsize=(8,10))

graphs = graphs.flatten()

for idx, graph in enumerate(graphs):

    graph.figure

    

    binwidth= (max(df[observe[idx]]) - min(df[observe[idx]]))/50

    bins = np.arange(min(df[observe[idx]]), max(df[observe[idx]]) + binwidth, binwidth)

    graph.hist([malignant[observe[idx]],benign[observe[idx]]], bins=bins, alpha=0.6, normed=True, label=['Malignant','Benign'], color=['red','blue'])

    graph.legend(loc='upper right')

    graph.set_title(observe[idx])

plt.tight_layout()
color_wheel = {0: "blue", 1: "red"}

colors = df["diagnosis"].map(lambda x: color_wheel.get(x))

pd.scatter_matrix(observables, c=colors, alpha = 0.5, figsize = (15, 15), diagonal = 'kde');
# Drop columns that do not aid in predicting type of cancer

observables.drop(['fractal_dimension_mean', 'smoothness_mean', 'symmetry_mean'],axis=1,inplace=True)
# Split data appropriately

X = observables

y = df['diagnosis']
gnb = GaussianNB()

gnb_scores = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')

print(gnb_scores.mean())
# Decide what k should be for KNN

knn = KNeighborsClassifier()



k_range = list(range(1, 30))

leaf_size = list(range(1,30))

weight_options = ['uniform', 'distance']

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options, 'algorithm': algorithm}
rand_knn = RandomizedSearchCV(knn, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)

rand_knn.fit(X,y)
print(rand_knn.best_score_)

print(rand_knn.best_params_)

print(rand_knn.best_estimator_)
dt_clf = DecisionTreeClassifier(random_state=42)



param_grid = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': sp_randint(2, 11), 

              'min_samples_leaf': sp_randint(1, 11)}
rand_dt = RandomizedSearchCV(dt_clf, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)

rand_dt.fit(X,y)
print(rand_dt.best_score_)

print(rand_dt.best_params_)

print(rand_dt.best_estimator_)
sv_clf = SVC(random_state=42)



param_grid = [

              {'C': [1, 10, 100, 1000], 

               'kernel': ['linear']

              },

              {'C': [1, 10, 100, 1000], 

               'gamma': [0.001, 0.0001], 

               'kernel': ['rbf']

              },

 ]
grid_sv = GridSearchCV(sv_clf, param_grid, cv=10, scoring="accuracy")

grid_sv.fit(X,y)
print(grid_sv.best_score_)

print(grid_sv.best_params_)

print(grid_sv.best_estimator_)
rf_clf = RandomForestClassifier(random_state=42)



param_grid = {"max_depth": [3, None],

              "max_features":  sp_randint(1, 8),

              "min_samples_split": sp_randint(2, 11),

              "min_samples_leaf": sp_randint(1, 11),

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}
rand_rf = RandomizedSearchCV(rf_clf, param_distributions=param_grid, n_iter=100, random_state=42)

rand_rf.fit(X,y)
print(rand_rf.best_score_)

print(rand_rf.best_params_)

print(rand_rf.best_estimator_)
# Using decision stumps due to size of sample.

# Attempting to prevent over-fitting

stump_clf =  DecisionTreeClassifier(random_state=42, max_depth=1)



param_grid = {

              "base_estimator__max_features": ['auto', 'sqrt', 'log2'],

              "n_estimators": list(range(1,500)),

              "learning_rate": np.linspace(0.01, 1, num=20),

             }
ada_clf = AdaBoostClassifier(base_estimator = stump_clf)



rand_ada = RandomizedSearchCV(ada_clf, param_grid, scoring = 'accuracy', n_iter=100, random_state=42)

rand_ada.fit(X,y)
print(rand_ada.best_score_)

print(rand_ada.best_params_)

print(rand_ada.best_estimator_)