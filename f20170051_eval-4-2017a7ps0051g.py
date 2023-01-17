import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE, SpectralEmbedding, Isomap

from sklearn.decomposition import KernelPCA, SparsePCA, IncrementalPCA, NMF, FastICA

import warnings

import skimage

from skimage import exposure

from skimage.feature import daisy

from skimage.feature import haar_like_feature

from skimage.feature import hog



warnings.filterwarnings("ignore")
# !pip install numpy==1.16.2
data = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

train, labels = [], []

for i in data:

#     image = skimage.transform.integral_image(i[1].mean(axis=2))

#     image = haar_like_feature(image, 0, 0, image.shape[0], image.shape[1])

    image = hog(i[1], orientations=9)

    image = daisy(i[1].mean(axis=2)).flatten()

    train.append(image)

    labels.append(i[0])

train = np.array(train)
from PIL import Image

i = np.random.randint(0,2275)

new_im = Image.fromarray(data[i][1])

new_im
labels[i]
X = train

n_features = X.shape[1]

n_samples = X.shape[0]



le = LabelEncoder()

le.fit(labels)

y = le.transform(labels)

target_names = np.unique(labels)

n_classes = 19



print("Total dataset size:")

print("n_samples: %d" % n_samples)

print("n_features: %d" % n_features)

print("n_classes: %d" % n_classes)
scaler = StandardScaler().fit(X)



# split into a training and testing set

# X = TSNE(n_components=2, init='pca').fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



n_components = 75



print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))



pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized').fit(X)

# pca = NMF(n_components=n_components).fit(X)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
# lda = LinearDiscriminantAnalysis(n_components=20)

# lda.fit(X_train_pca,y_train)

# X_train_pca = lda.transform(X_train_pca)

# X_test_pca = lda.transform(X_test_pca)
print("Fitting the classifier to the training set")



param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 

              'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 

              'kernel': ['rbf','sigmoid','linear','poly'] }

# clf = LinearSVC()

# clf = SGDClassifier() //lol

# clf = KNeighborsClassifier()  //okay

# clf = SGDClassifier()  //nope

# clf = LogisticRegression() //better

# clf = RandomForestClassifier() //okay

# clf = AdaBoostClassifier() //slow and nope

# clf = ExtraTreesClassifier() //disappointing

# clf = GradientBoostingClassifier() //same



# clf = SVC()

# params from Grid Search

clf = SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,

          decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf', max_iter=-1, 

          probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)



# clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, iid=False, scoring='f1_weighted')



# clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

clf = clf.fit(X_train_pca, y_train)
# print("Best estimator found by grid search:")

# print(clf.best_estimator_)
print("Predicting people's names on the test set")

y_pred = clf.predict(X_test_pca)



print(classification_report(y_test, y_pred, target_names=target_names))

print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# #############################################################################

# Qualitative evaluation of the predictions using matplotlib



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):

    """Helper function to plot a gallery of portraits"""

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):

        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(images[i].reshape((h, w, 3)), cmap=plt.cm.gray)

        plt.title(titles[i], size=12)

        plt.xticks(())

        plt.yticks(())

#         print('\n')





# # plot the result of the prediction on a portion of the test set



# def title(y_pred, y_test, target_names, i):

#     pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]

#     true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]

#     return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)



# prediction_titles = [title(y_pred, y_test, target_names, i)

#                      for i in range(y_pred.shape[0])]



plot_gallery(X, prediction_titles, 50, 50)



# plot the gallery of the most significative eigenfaces



eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

#plot_gallery(scaler.inverse_transform(eigenfaces.reshape((120, 2500))), eigenface_titles, 50, 50)



plt.show()
import umap.umap_ as umap
embedding = reducer.fit_transform(X)
t0 = time()

reducer = SpectralEmbedding(n_components=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train = reducer.fit_transform(X_train, y_train)

# clf_test = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,

#                decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf', 

#                max_iter=-1, probability=False, random_state=None, shrinking=True, 

#                tol=0.001, verbose=False)



param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='sigmoid', class_weight='balanced'),

                   param_grid, cv=5, iid=False, scoring='f1_weighted')



clf.fit(X_train, y_train)

clf_best = clf.best_estimator_

y_pred = clf_best.predict(reducer.fit_transform(X_test))



print(classification_report(y_test, y_pred, target_names=target_names))

print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
train
# generate test labels

test_data = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)

test, id = [], []

for i in test_data:

#     image = skimage.transform.integral_image(i[1].mean(axis=2))

#     image = haar_like_feature(image, 0, 0, image.shape[0], image.shape[1])

    image = hog(i[1], orientations=9)

    image = daisy(i[1].mean(axis=2)).flatten()

    test.append(image)

    id.append(i[0])

test = scaler.transform(np.array(test))

test_pca = pca.transform(test)

test_pred = clf.predict(test_pca)

test_labels = le.inverse_transform(test_pred)

submit = pd.concat([pd.Series(id), pd.Series(test_labels)], axis=1)

submit.columns = ['ImageId', 'Celebrity']

submit.to_csv('/kaggle/working/submission.csv', index=False)
# from sklearn.decomposition import KernelPCA
train