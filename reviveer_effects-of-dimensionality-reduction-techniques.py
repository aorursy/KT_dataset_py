# data manipulation

import pandas as pd

import numpy as np



# sklearn helper functions

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_validate



# Dimensionality Reduction algorithms

from sklearn.decomposition import PCA, KernelPCA

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE

from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# sklearn ML algorithms

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression



# Viz

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnnotationBbox, OffsetImage



# Hide warnings

import warnings

warnings.filterwarnings('ignore')
# read train set file

df_train_raw = pd.read_csv("../input/digit-recognizer/train.csv")



# check dimensions of train set

print("Train Set has", df_train_raw.shape[0], "rows and", df_train_raw.shape[1], "columns.")
# splitting method which keeps 20% data for validation set

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in sss.split(df_train_raw, df_train_raw['label']):

    strat_train_set = df_train_raw.loc[train_index]

    df_validation = df_train_raw.loc[test_index]

    

# creating a copy of target labels and removing them from validation set

validation_labels = df_validation['label'].copy()

df_validation.drop(columns=['label'], inplace=True)
scaler = StandardScaler()    # Standardize features by removing the mean and scaling to unit variance

df_validation = scaler.fit_transform(df_validation)
# custom cross-validation for comparing multiple estimators

def cross_validation(data, target_labels, mla_list, split_method):

    

    MLA_columns = ['MLA Name', 'MLA Parameters', 'Train Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy 3*STD', 'Training Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)



    row_index = 0

    for alg in mla_list:



        # set name and parameters

        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())



        # cross validation

        cv_results = cross_validate(alg, data, target_labels, cv=split_method, scoring=['accuracy'], return_train_score=True)



        MLA_compare.loc[row_index, 'Training Time'] = cv_results['fit_time'].sum()

        MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_accuracy'].mean()

        MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_accuracy'].mean()

        MLA_compare.loc[row_index, 'Test Accuracy 3*STD'] = cv_results['test_accuracy'].std()*3



        row_index+=1



    # print and sort table

    MLA_compare.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)

    

    return MLA_compare
# list of classifiers we want to test

mla_list = [

    RandomForestClassifier(n_jobs=-1),

    

    SGDClassifier(early_stopping=True, n_iter_no_change=5, n_jobs=-1),

    

    LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, n_jobs=-1),    # Softmax Regression

]



# splitting method for use in 'cross_validate'

cv_split_method = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)
cross_validation(data=df_validation, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
n_components = 100
pca = PCA(n_components=n_components)



df_validation_pca = df_validation.copy()

df_validation_pca = pca.fit_transform(df_validation_pca)
cross_validation(data=df_validation_pca, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
kpca = KernelPCA(n_components=n_components, kernel='rbf')



df_validation_kpca = df_validation.copy()

df_validation_kpca = kpca.fit_transform(df_validation_kpca)
cross_validation(data=df_validation_kpca, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
# multiple distortion values will give multiple components value

johnson_lindenstrauss_min_dim(df_validation.shape[0], eps=[0.5, 0.6, 0.7, 0.8, 0.9])
grp = GaussianRandomProjection(n_components='auto', eps=0.9, random_state=0)



df_validation_grp = df_validation.copy()

df_validation_grp = grp.fit_transform(df_validation_grp)
cross_validation(data=df_validation_grp, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
srp = SparseRandomProjection(n_components='auto', eps=0.9, random_state=0)



df_validation_srp = df_validation.copy()

df_validation_srp = srp.fit_transform(df_validation_srp)
cross_validation(data=df_validation_srp, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=10)



df_validation_lle = df_validation.copy()

df_validation_lle = lle.fit_transform(df_validation_lle)
cross_validation(data=df_validation_lle, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
iso = Isomap(n_components=n_components, n_neighbors=10, n_jobs=-1)



df_validation_iso = df_validation.copy()

df_validation_iso = iso.fit_transform(df_validation_iso)
cross_validation(data=df_validation_iso, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)



df_validation_tsne = df_validation.copy()

df_validation_tsne = tsne.fit_transform(df_validation_tsne)
cross_validation(data=df_validation_tsne, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
lda = LinearDiscriminantAnalysis(n_components=9)



df_validation_lda = df_validation.copy()

df_validation_lda = lda.fit_transform(df_validation_lda, validation_labels)
cross_validation(data=df_validation_lda, target_labels=validation_labels, mla_list=mla_list, split_method=cv_split_method)
# using only 2 components

pca = PCA(n_components=2)

kpca = KernelPCA(n_components=2, kernel='rbf')

grp = GaussianRandomProjection(n_components=2, random_state=0)

srp = SparseRandomProjection(n_components=2, random_state=0)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)

mds = MDS(n_components=2, max_iter=2, random_state=0)

iso = Isomap(n_components=2, n_neighbors=10)

lda = LinearDiscriminantAnalysis(n_components=2)



# transform data for vizualization

df_reduced_pca = df_validation.copy()

df_reduced_pca = pca.fit_transform(df_reduced_pca)



df_reduced_kpca = df_validation.copy()

df_reduced_kpca = kpca.fit_transform(df_reduced_kpca)



df_reduced_grp = df_validation.copy()

df_reduced_grp = grp.fit_transform(df_reduced_grp)



df_reduced_srp = df_validation.copy()

df_reduced_srp = srp.fit_transform(df_reduced_srp)



df_reduced_lle = df_validation.copy()

df_reduced_lle = lle.fit_transform(df_reduced_lle)



df_reduced_mds = df_validation.copy()

df_reduced_mds = mds.fit_transform(df_reduced_mds)



df_reduced_iso = df_validation.copy()

df_reduced_iso = iso.fit_transform(df_reduced_iso)



df_reduced_tsne = df_validation_tsne



df_reduced_lda = df_validation.copy()

df_reduced_lda = lda.fit_transform(df_reduced_lda, validation_labels)
titles = ["PCA", "KernelPCA", "Gaussian Random Projection", "Sparse Random Projection", "LLE", "MDS", "Isomap", "t-SNE", "LDA"]



fig, axs = plt.subplots(3, 3, figsize=(20,20))



axs = axs.ravel()



for subplot, title, df_reduced in zip(axs, titles, (df_reduced_pca, df_reduced_kpca, df_reduced_grp, df_reduced_srp, df_reduced_lle, df_reduced_mds, df_reduced_iso, df_reduced_tsne, df_reduced_lda)):

    plt.subplot(subplot)

    plt.title(title, fontsize=14)

    plt.scatter(df_reduced[:, 0], df_reduced[:, 1], c=validation_labels, cmap=plt.cm.hot)

    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 101:

        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    plt.grid(True)



plt.show()
plt.figure(figsize=(13,10))

plt.scatter(df_reduced_tsne[:, 0], df_reduced_tsne[:, 1], c=validation_labels, cmap="jet")

plt.axis('off')

plt.colorbar()

plt.show()
def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):

    

    # scale the input features so that they range from 0 to 1

    X_normalized = MinMaxScaler().fit_transform(X)

    

    # Now we create the list of coordinates of the digits plotted so far.

    # We pretend that one is already plotted far away at the start, to

    # avoid `if` statements in the loop below

    neighbors = np.array([[10., 10.]])

    

    plt.figure(figsize=figsize)

    cmap = mpl.cm.get_cmap("jet")

    digits = np.unique(y)

    

    for digit in digits:

        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])

    

    plt.axis("off")

    ax = plt.gcf().gca()  # get current axes in current figure

    

    for index, image_coord in enumerate(X_normalized):

        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()

        if closest_distance > min_distance:

            neighbors = np.r_[neighbors, [image_coord]]

            if images is None:

                plt.text(image_coord[0], image_coord[1], str(int(y[index])),

                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})

            else:

                image = images[index].reshape(28, 28)

                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)

                ax.add_artist(imagebox)
plot_digits(df_reduced_tsne, validation_labels, images=df_validation, figsize=(35,25))
from sklearn.datasets import make_swiss_roll

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)



axes = [-11.5, 14, -2, 23, -12, 15]



x2s = np.linspace(axes[2], axes[3], 10)

x3s = np.linspace(axes[4], axes[5], 10)

x2, x3 = np.meshgrid(x2s, x3s)



fig = plt.figure(figsize=(6, 5))

ax = plt.subplot(111, projection='3d')



positive_class = X[:, 0] > 5

X_pos = X[positive_class]

X_neg = X[~positive_class]

ax.view_init(10, -70)

ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")

ax.plot_wireframe(5, x2, x3, alpha=0.5)

ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



plt.show()



fig = plt.figure(figsize=(5, 4))

ax = plt.subplot(111)



plt.plot(t[positive_class], X[positive_class, 1], "gs")

plt.plot(t[~positive_class], X[~positive_class, 1], "y^")

plt.axis([4, 15, axes[2], axes[3]])

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.grid(True)



plt.show()



fig = plt.figure(figsize=(6, 5))

ax = plt.subplot(111, projection='3d')



positive_class = 2 * (t[:] - 4) > X[:, 1]

X_pos = X[positive_class]

X_neg = X[~positive_class]

ax.view_init(10, -70)

ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")

ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



plt.show()



fig = plt.figure(figsize=(5, 4))

ax = plt.subplot(111)



plt.plot(t[positive_class], X[positive_class, 1], "gs")

plt.plot(t[~positive_class], X[~positive_class, 1], "y^")

plt.plot([4, 15], [0, 22], "b-", linewidth=2)

plt.axis([4, 15, axes[2], axes[3]])

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.grid(True)



plt.show()