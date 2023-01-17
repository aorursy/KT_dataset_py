import numpy as np



from sklearn.datasets import load_digits

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold.t_sne import _joint_probabilities

from sklearn.model_selection import train_test_split

from sklearn.metrics import pairwise_distances

from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeClassifier



from scipy.spatial.distance import pdist

from scipy import linalg

from scipy.spatial.distance import squareform



from matplotlib import pyplot as plt



import seaborn as sns



from time import time



sns.set(rc={'figure.figsize':(11.7,8.27)})

palette = sns.color_palette("bright", 10)



seed = 19960614

np.random.seed(seed)
X, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=seed)



dt = DecisionTreeClassifier(max_depth=10, random_state=seed)



t0 = time()

dt.fit(x_train, y_train)

t1 = time()



print('Time spent training the model: {} s'.format(round(t1-t0, 2)))



resultado_sin_tsne = dt.predict(x_test)



from sklearn.metrics import accuracy_score



print('Accuracy without tsne nor tsvd: {}'.format(round(accuracy_score(y_test, resultado_sin_tsne) * 100, 2)))
X, y = load_digits(return_X_y=True)



tsne = TSNE()



t0 = time()

tsned_images = tsne.fit_transform(X)

t1 = time()

print('Time spent converting the data: {} s'.format(round(t1-t0,2)))



dt = DecisionTreeClassifier(random_state=seed)



X_train, X_test, y_train, y_test = train_test_split(tsned_images, y, random_state=seed)



sns.scatterplot(X_train[:,0], X_train[:,1], hue=y_train, legend='full', palette=palette)

plt.show()



t2 = time()

dt.fit(X_train, y_train)

t3 = time()



print('Time spent training the model: {} s'.format(round(t2-t3,2)))



result = dt.predict(X_test)



print('Accuracy with tsne: {}'.format(round(accuracy_score(y_test, result) * 100, 2)))
X, y = load_digits(return_X_y=True)





t0 = time()

tsvd_images = TruncatedSVD(n_components=50).fit_transform(X)

t1 = time()

print('Time spent converting the raw data with tsvd: {} s'.format(round(t1-t0,2)))



tsne = TSNE()



t2 = time()

tsvd_tsne_images = tsne.fit_transform(tsvd_images)

t3 = time()

print('Time spent converting the tsvd data with tsne: {} s'.format(round(t3-t2,2)))



sns.scatterplot(tsvd_tsne_images[:,0], tsvd_tsne_images[:,1], hue=y, legend='full', palette=palette)



plt.show()



dt = DecisionTreeClassifier(random_state=seed)



X_train, X_test, y_train, y_test = train_test_split(tsvd_tsne_images, y, random_state=seed)



t4 = time()

dt.fit(X_train, y_train)

t5 = time()



print('Time spent training the model: {} s'.format(round(t5-t4,2)))



resultado_con_tsne = dt.predict(X_test)



print('Accuracy with tsvd and tsne: {}'.format(round(accuracy_score(y_test, resultado_con_tsne) * 100, 2)))