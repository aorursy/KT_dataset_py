import pandas as pd

import numpy as np



import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline
train_images = pd.read_csv('../input/train.csv')

test_images = pd.read_csv('../input/test.csv')
train_images.head()
X = (train_images.drop(['label'], axis=1).values) / 255

y = (train_images['label'].values)

X_final_test = (test_images.values) / 255



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
image = X[1]

image =image.reshape((28,28))

plt.imshow(image,cmap='gray')

plt.title(y[1])
clf = svm.SVC()

clf.fit(X_train[0:5000], y_train[0:5000])

clf.score(X_test[0:5000], y_test[0:5000])
# X_test[X_test>0]=1

# X_train[X_train>0]=1



img=X_train[1].reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(y_train[1])
plt.hist(X_train[1])
from sklearn.decomposition import PCA



pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

X_final_test_pca = pca.transform(X_final_test)
from sklearn.model_selection import GridSearchCV



parameter_grid = {

    'C': [1, 0.9, 0.8, 0.7, 0.6, 0.1, 0.01],

    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']

}



grid_search = GridSearchCV(svm.SVC(), parameter_grid, cv=5, verbose=3)
grid_search.fit(X_train_pca[0:5000], y_train[0:5000])
clf = svm.SVC()

clf.fit(X_train_pca[0:5000], y_train[0:5000])



print(clf.score(X_train_pca[0:5000], y_train[0:5000]))

print(clf.score(X_test_pca[0:5000], y_test[0:5000]))
print(grid_search.score(X_train_pca[0:5000], y_train[0:5000]))

print(grid_search.score(X_test_pca[0:5000], y_test[0:5000]))
results = grid_search.predict(X_final_test_pca)
results
df = pd.DataFrame(results)
df.index+=1

df.index.rename('ImageId', inplace=True)
df.columns=['Label']

df.to_csv('submission_results.csv', header=True)
df