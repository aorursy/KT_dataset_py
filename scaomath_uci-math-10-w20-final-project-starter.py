import numpy as np

import matplotlib.pyplot as plt
!ls ../input
# Load the data



# the images for training

X_train = np.load('../input/uci-math-10-winter2020/kmnist-train-imgs.npz')['X'] 



# the labels (category) for images 

y_train = np.load('../input/uci-math-10-winter2020/kmnist-train-labels.npz')['y']



# the images for competition (you want to use your model to predict the labels)

X_test = np.load('../input/uci-math-10-winter2020/kmnist-test-imgs.npz')['X'] 
# verifying the shapes

print(f"The shapes of train, test set are {X_train.shape}, {X_test.shape}.")
fig, axes = plt.subplots(4,5, figsize=(12, 12))

axes = axes.reshape(-1)

np.random.seed(1)

idx = np.random.choice(X_train.shape[0], size=20)



for i in range(20):

    axes[i].axis('off') # hide the axes ticks

    axes[i].imshow(X_train[idx[i]], cmap = 'gray')

    axes[i].set_title(str(y_train[idx[i]]), color= 'black', fontsize=25)

plt.show()
# Flatten images

X_train = np.reshape(X_train, (-1, 784))

X_test = np.reshape(X_test, (-1, 784))

print(f"Shapes: {X_train.shape}, {X_test.shape}")
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) # knn should take a long time
solutions = np.zeros((X_test.shape[0], 2))

solutions[:,0] = np.arange(1,X_test.shape[0]+1)

solutions[:,1] = y_pred

solutions = solutions.astype(int)

np.savetxt("solutions-yournames.csv", solutions, 

           fmt='%s', header = 'Id,Category', delimiter = ',', comments='')