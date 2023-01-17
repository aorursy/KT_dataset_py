import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['image.cmap'] = 'gray'
def read_vectors(filename):
    return np.fromfile(filename, dtype=np.uint8).reshape(-1,401)
data_train = np.vstack(tuple(read_vectors("../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn))
                      for nn in range(2)))
y_train = data_train[:,0]
X_train = data_train[:,1:]

data_test = np.vstack(tuple(read_vectors("../input/snake-eyes/snakeeyes_test.dat")))
y_test = data_test[:,0]
X_test = data_test[:,1:]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
idx_two = y_train>6
y_two = [2]*sum(idx_two)
X_two = X_train[idx_two]

idx_one = y_train==1
y_one = [1]*sum(idx_one)
X_one = X_train[idx_one]

X_number = np.concatenate((X_two, X_one))
y_number = np.append(y_two, y_one)
X_train_dices, X_test_dices, y_train_dices, y_test_dices = train_test_split(X_number, 
                                                            y_number, test_size=0.5)
rf = RandomForestClassifier()
rf.fit(X_train_dices, y_train_dices)
y_pred_dices = rf.predict(X_test_dices)
print(accuracy_score(y_test_dices, y_pred_dices))
faces_train = rf.predict(X_train)
faces_test = rf.predict(X_test)
plt.figure(figsize=(20,10))
for k in range(1,61):
    plt.subplot(6, 10, k)
    plt.imshow(X_train[k-1].reshape(20,20))
    plt.title(faces_train[k-1])
    plt.axis('off')
    plt.axis('off')

X_train_1 = X_train[faces_train==1]
y_train_1 = y_train[faces_train==1]

X_train_2 = X_train[faces_train==2]
y_train_2 = y_train[faces_train==2]

X_test_1 = X_test[faces_test==1]
y_test_1 = y_test[faces_test==1]

X_test_2 = X_test[faces_test==2]
y_test_2 = y_test[faces_test==2]
from sklearn.decomposition import PCA
pca = PCA(random_state=0, whiten=True)
pca.fit(X_train_1)
exp_var_cum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(exp_var_cum.size), exp_var_cum)
plt.grid()
pca = PCA(n_components=150, random_state=0, whiten=True)
pca.fit(X_train_2)
X_train_pca = pca.transform(X_train_2)
X_test_pca = pca.transform(X_test)
X_reconstructed_pca = pca.inverse_transform(X_test_pca)

plt.figure(figsize=(20,10))
for k in range(20):
    plt.subplot(4, 10, k*2 + 1)
    plt.imshow(X_test[k].reshape(20,20))
    plt.axis('off')
    plt.subplot(4, 10, k*2 + 2)
    plt.imshow(X_reconstructed_pca[k].reshape(20,20))
    plt.axis('off')
pca = PCA(n_components=150, random_state=0, whiten=True)
pca.fit(X_train_1)
X_train_pca = pca.transform(X_train_1)
X_test_pca = pca.transform(X_test)
plt.figure(figsize=(20,10))
for k in range(40):
    plt.subplot(4,10,k+1)
    plt.imshow(pca.components_[k].reshape(20,20))
    plt.axis('off')
pca = PCA(n_components=150, random_state=0, whiten=True)
pca.fit(X_train_2)
X_train_pca = pca.transform(X_train_2)
X_test_pca = pca.transform(X_test)
plt.figure(figsize=(20,10))
for k in range(40):
    plt.subplot(4,10,k+1)
    plt.imshow(pca.components_[k].reshape(20,20))
    plt.axis('off')
rf = RandomForestClassifier()
rf.fit(X_train_1, y_train_1)
pred = rf.predict(X_test_1)
accuracy_score(pred, y_test_1)
rf = RandomForestClassifier()
rf.fit(X_train_2, y_train_2)
pred = rf.predict(X_test_2)
accuracy_score(pred, y_test_2)

