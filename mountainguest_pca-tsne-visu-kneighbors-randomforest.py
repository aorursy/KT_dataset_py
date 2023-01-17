import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')   #probably won't be used here
train.shape, test.shape
margin = slice(2,66)
shape = slice(66,130)
texture = slice(130,None)
species = slice(1,1)
from matplotlib import cm
colmap = cm.get_cmap('cool', 30)
train.head(3)
classes = train["species"].value_counts()

count=len(classes)
avg=classes[:].mean()
std=classes[:].std()

print("{0} classes with {1} (+/-{2}) instances per class,".format(count, avg, std))
print("{0} training instances with {1} features.".format(len(train), train.shape[1]))
def corr_sub_plot(ax, df, title=""):
    corr = df.corr()
    
    avg_corr = corr.values[np.triu_indices_from(corr.values,1)].mean()
    ax.set_title(title+" ({0:.4})".format(avg_corr))
    labels=range(1,len(corr.columns),4)
    ax.set_yticks(labels)
    ax.set_yticklabels(labels)
    return ax.imshow(corr, interpolation="nearest", cmap=colmap, vmin=-1, vmax=1)


f, ax = plt.subplots(2, 2,figsize=(10,10))

corr_sub_plot(ax[0,0], train.iloc[:,margin], title="Margin")
corr_sub_plot(ax[0,1], train.iloc[:,shape], "Shape")
cax = corr_sub_plot(ax[1,0], train.iloc[:,texture], "Texture")

f.colorbar(cax, ax=ax.ravel().tolist())

ax[1,1].set_visible(False)
from sklearn.preprocessing import StandardScaler

texture_n = StandardScaler().fit_transform(train.iloc[:,texture])
shape_n = StandardScaler().fit_transform(train.iloc[:,shape])
margin_n = StandardScaler().fit_transform(train.iloc[:,margin])
from sklearn.decomposition import PCA

def pca_variance(data, keeped_variance):
    pca = PCA(n_components=keeped_variance)
    proj_margin =pca.fit_transform(data)
    return pca.n_components_
pca_red = "PCA reduced 65 features to {0}, preserving {1}% of the input's variance "
print(pca_red.format(pca_variance(texture_n, 0.95), 0.95))
print(pca_red.format(pca_variance(texture_n, 0.99), 0.99))
ranger = np.arange(0.90, 1, 0.005)
dims_texture = [pca_variance(texture_n, e) for e in ranger]
dims_margin = [pca_variance(margin_n, e) for e in ranger]
dims_shape = [pca_variance(shape_n, e) for e in ranger]

plt.plot(ranger, dims_texture, label="texture")
plt.plot(ranger, dims_margin, label="margin")
plt.plot(ranger, dims_shape, label="shape")
plt.legend(loc='upper left')
from sklearn.manifold import TSNE

pre_X = np.concatenate([texture_n, margin_n, shape_n], axis=1)
X_reduced = TSNE(n_components=2, random_state=4422, init="pca").fit_transform(pre_X)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# cat = CategoricalEncoder(encoding="ordinal")  # waiting next sklearn update
labels = le.fit_transform(train.iloc[:, 1])
plt.figure(figsize=(10,10))

for i in range(0,99):
    keeped= labels == i
    plt.plot(X_reduced[keeped, 0], X_reduced[keeped, 1], linestyle=':')

plt.axis('off')
X_reduced_3D = TSNE(n_components=3, random_state=4422).fit_transform(pre_X)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)

#in order to interact with 3D plots
#%matplotlib notebook 

for i in range(0,99):
    keeped= labels == i
    ax.plot(X_reduced_3D[keeped, 0], X_reduced_3D[keeped, 1], X_reduced_3D[keeped, 2], linestyle=':')

ax.view_init(20, 35)

import matplotlib.image as mimg

plt.figure(1)
img = mimg.imread('../input/images/99.jpg')
plt.imshow(img, cmap='cool')
from skimage.feature import *
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure


fig, ax = plt.subplots()

# 130, 12 are nice
img_to_process = ndimage.binary_erosion(mimg.imread('../input/images/90.jpg'), structure=np.ones((2,2)))

center_of_mass = ndimage.measurements.center_of_mass(img_to_process)

coords = corner_peaks(corner_harris(img_to_process, k=0), min_distance=4)
coords_subpix = corner_subpix(img_to_process, coords, alpha=0.2)

ax.imshow(img_to_process, interpolation='nearest', cmap='cool')

# the extraction of sharp edges is still in progress
# ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+w', markersize=15) 
ax.plot(center_of_mass[1], center_of_mass[0],'og', markersize=10)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin

class SliceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, slice):
        self.slices = slice
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.iloc[:,self.slices]
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

def make_num_pipeline(slices, pca_coef):
    return Pipeline([('cursor', SliceSelector(slices)), 
                     ('scaler', StandardScaler()), 
                     ('PCA', PCA(n_components=pca_coef)),])

margin_pipeline = make_num_pipeline(margin, 0.96)
shape_pipeline = make_num_pipeline(shape, 0.98)
texture_pipeline = make_num_pipeline(texture, 0.96)
full_pipeline = FeatureUnion(transformer_list=[
    ("margin_pipeline", margin_pipeline),
    ("shape_pipeline", shape_pipeline),
    ("texture_pipeline", texture_pipeline),
])
y = labels
X = full_pipeline.fit_transform(train)
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index],  y[test_index]
from sklearn.metrics import accuracy_score, log_loss
def fast_tests(clf, X, y, power=1):
    preds = clf.predict(X_test)
    acc = accuracy_score(y, preds)
    logloss = log_loss(y_test, np.power(clf.predict_proba(X_test), power), labels=y)
    return acc, logloss
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(weights="distance",n_jobs=-1, p=2)
neigh.fit(X_train, y_train) 
fast_tests(neigh, X_test, y_test)
param_dist = {"n_neighbors": range(4,10),
              "weights":["distance"],
             "leaf_size": range(1,10),
             }
rnd_n = RandomizedSearchCV(KNeighborsClassifier(p=2), param_dist, n_jobs=-1, n_iter=20)

rnd_n.fit(X_train, y_train) 
optimized_neigh = rnd_n.best_estimator_
print(optimized_neigh)
fast_tests(optimized_neigh, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=28, n_estimators=500,random_state=0, max_features=20, n_jobs=-1)
forest.fit(X_train, y_train)
fast_tests(forest, X_test, y_test)
from scipy.stats import randint as sp_randint
param_dist = {"max_depth": sp_randint(20, 30),
              "max_features": sp_randint(15, 60),
              "min_samples_split": sp_randint(2, 10),
             }
forest=RandomForestClassifier(n_jobs=-1, random_state=4422, bootstrap=True, criterion="entropy", n_estimators=600)
rnd_f = RandomizedSearchCV(forest, param_dist, n_iter=15, n_jobs=-1)

rnd_f.fit(X_train, y_train) 
optimized_forest = rnd_f.best_estimator_
print(optimized_forest)
optimized_forest.fit(X_train, y_train) 
fast_tests(optimized_forest, X_test, y_test)
from sklearn.ensemble import VotingClassifier

ests = [('neigh', optimized_neigh), ('forest', optimized_forest)]
voting_clf = VotingClassifier(estimators=ests, voting='soft')
voting_clf.fit(X_train, y_train)
fast_tests(voting_clf, X_test, y_test)