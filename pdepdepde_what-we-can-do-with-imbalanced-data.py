# ! pip install -U imbalanced-learn

# ! pip install mlxtend
import itertools



from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification



from mlxtend.plotting import plot_decision_regions



import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt



%matplotlib inline
svm = LinearSVC(random_state=2019)

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(20,20))



class_weights = [[.3,  .3,   .3], 

                 [.7,  .2,   .1],

                 [.93, .05,  .02],

                 [.97, .017, .13]]



for _weights, grd in zip(class_weights, itertools.product([0, 1], repeat=2)):

    

    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,

                           n_redundant=0, n_repeated=0, n_classes=3,

                           n_clusters_per_class=1,

                            weights=_weights,

                            class_sep=0.8, random_state=42)

    svm.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=svm, legend=2)

    plt.title(f'SVM with {_weights} class proportion');
from imblearn.over_sampling import  RandomOverSampler, SMOTE, ADASYN



svm = LinearSVC(random_state=2019)

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(20,20))



X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,

                           n_redundant=0, n_repeated=0, n_classes=3,

                           n_clusters_per_class=1, weights=[0.03, 0.07, 0.9],

                           class_sep=0.8, random_state=1)





resamplers = [ RandomOverSampler(random_state=42), 

              SMOTE(random_state=42), 

              ADASYN(random_state=42) ]



titles = ['RandomOverSampler', 'SMOTE', 'ADASYN']



for resampler, grd, title in zip(resamplers, itertools.product([0, 1], repeat=2), titles):

    X_resampled, y_resampled = resampler.fit_resample(X, y)

    svm.fit(X_resampled, y_resampled)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X_resampled, y=y_resampled, clf=svm, legend=2)

    plt.title(f'SVM anbd {title} method');



svm.fit(X, y)

ax = plt.subplot(gs[1, 1])

fig = plot_decision_regions(X=X, y=y, clf=svm, legend=2)

plt.title(f'SVM and NO method');

from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss



svm = LinearSVC(random_state=2019)

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(20,20))



X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,

                           n_redundant=0, n_repeated=0, n_classes=3,

                           n_clusters_per_class=1, weights=[0.03, 0.07, 0.9],

                           class_sep=0.8, random_state=1)





resamplers = [ RandomUnderSampler(random_state=42), 

              ClusterCentroids(random_state=42), 

              NearMiss(random_state=42) ]



titles = ['RandomUnderSampler', 'ClusterCentroids', 'NearMiss']



for resampler, grd, title in zip(resamplers, itertools.product([0, 1], repeat=2), titles):

    X_resampled, y_resampled = resampler.fit_resample(X, y)

    svm.fit(X_resampled, y_resampled)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X_resampled, y=y_resampled, clf=svm, legend=2)

    plt.title(f'SVM anbd {title} method');



svm.fit(X, y)

ax = plt.subplot(gs[1, 1])

fig = plot_decision_regions(X=X, y=y, clf=svm, legend=2)

plt.title(f'SVM and NO method');