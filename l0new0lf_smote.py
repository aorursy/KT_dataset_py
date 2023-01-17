import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



np.random.seed(123)
import sklearn

from sklearn.datasets import make_classification



X, y = make_classification(

            n_samples=10000, 

            n_features=2,

            n_redundant=0,

            n_classes=2,

            n_clusters_per_class=1, 

            weights=[0.99], 

            flip_y=0, 

            random_state=1

)
plt.scatter(X[:,0], X[:,1], c=y)

plt.show()
labels, cnts = np.unique(y, return_counts=True)



for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

_X, _y = oversample.fit_resample(X, y)
plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()
labels, cnts = np.unique(_y, return_counts=True)



for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.under_sampling import RandomUnderSampler # undersample majority (wrt minority)

from imblearn.over_sampling import SMOTE # oversample minority (wrt majority)
# OVERSAMPLE MINORITY

# ===================



oversample = SMOTE(sampling_strategy=0.1)

_X, _y = oversample.fit_resample(X, y)
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
# UNDERSAMPLE MAJORITY

# ====================



undersample = RandomUnderSampler(sampling_strategy=0.5)

_X, _y = undersample.fit_resample(X, y)
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.pipeline import Pipeline



over = SMOTE(sampling_strategy=0.1)

under = RandomUnderSampler(sampling_strategy=0.5) # huge loss in majority data

steps = [('oversample_minority', over), ('undersample_majority', under)]



pipeline = Pipeline(steps=steps)
_X, _y = pipeline.fit_resample(X, y)



# first increase minority

# then decrease majority such that `majority_new = 2 * minority`
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.over_sampling import BorderlineSMOTE
# Oversample Selectively where Mistakes are High

# ===============================================

sel_over = BorderlineSMOTE()

_X, _y = sel_over.fit_resample(X, y)
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.over_sampling import SVMSMOTE
# Oversample Selectively where Mistakes are High

# using SVM instead

# ===============================================

sel_over = SVMSMOTE()

_X, _y = sel_over.fit_resample(X, y)
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")
from imblearn.over_sampling import ADASYN
# Oversample Selectively where Desnilty is Low

# ===============================================

sel_over = ADASYN()

_X, _y = sel_over.fit_resample(X, y)
# plot

# ----

plt.scatter(_X[:,0], _X[:,1], c=_y)

plt.show()



# print lables count

# ------------------

labels, cnts = np.unique(_y, return_counts=True)

for label, cnt in zip(labels, cnts):

    print(f"{label}\t: {cnt}")