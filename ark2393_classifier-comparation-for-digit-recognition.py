import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.gridspec as gridspec #to manipulate subplots grids

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import preprocessing, manifold, svm, tree, linear_model, metrics, ensemble

from sklearn.decomposition import PCA
df_digits = pd.read_csv('../input/train.csv')

df_digits.info()
df_digits.head()
img = df_digits.iloc[1][1:].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(df_digits.iloc[1,0])
f, axarr = plt.subplots(nrows = 3, ncols=3, sharey = True, sharex= True)

i = 0

for ir in range(3):

    for ic in range(3):

        img = df_digits.iloc[i][1:].as_matrix()

        img = img.reshape((28,28))

        axarr[ir,ic].imshow(img,cmap = 'gray')

        axarr[ir,ic].set_title(df_digits.iloc[i,0])

        i+=1

f.subplots_adjust(hspace = 0.5)
df_features = df_digits.iloc[:,1:]

digits_labels = df_digits['label']
plt.hist(digits_labels)
plt.hist(df_features.max())
f, axarr = plt.subplots(2,5, figsize = (9,9))

gs1 = gridspec.GridSpec(2,5)

i=0

for ir in range(2):

    for ic in range(5):

        temp = df_features[digits_labels==i]   

        axarr[ir,ic].imshow(temp.mean().reshape(28,28))

        i+=1

f.subplots_adjust(hspace = -0.5)
count, bins= np.histogram(df_features.mean(),bins = 14)

max(bins)
[x for x in zip(count, bins[1:])]
plt.plot(bins[1:]-5, count, 'ro-')

plt.title('Mean Pixel\'s Value Count')

plt.xlabel('Mean Pixel Value')

plt.ylabel('Count')
sum(count[-2:])
n_components = 35
sc = preprocessing.StandardScaler()

df_features_std = sc.fit_transform(df_features)

x_train, x_test, y_train, y_test = train_test_split(df_features_std, digits_labels, test_size = 0.25)
pca_ = PCA(n_components=n_components)

pca_.fit(df_features, digits_labels)

eigen_val = pca_.components_.reshape(n_components, 28,28)
n_row = 5

n_col = 7

fig = plt.figure(figsize=(8,9))

for i in range(n_row * n_col):

    offset =0

    plt.subplot(n_row, n_col, i + 1)

    plt.imshow(eigen_val[i].reshape(28,28), cmap='jet')

    title_text = 'Eigenvalue ' + str(i + 1)

    plt.title(title_text, size=6.5)

    plt.xticks(())

    plt.yticks(())

plt.show()
n_components_ = 7
from random import randint

colors = []

for i in range(10):

    colors.append('%06X' % randint(0, 0xFFFFFF))
pca_ = PCA(n_components=n_components_)

pca_features = pca_.fit_transform(x_train)

fig = plt.figure(figsize= (12,12))
fig = plt.figure(figsize=(12,8))

for i in range(10):

    to_plot = pca_features[:2000,:2][y_train[:2000].values==i]

    plt.scatter(to_plot[:,0],to_plot[:,1], marker='.', cmap=colors[i])

    plt.grid(b='on')
clfs = [tree.DecisionTreeClassifier(), linear_model.LogisticRegression(),

        svm.SVC(C=1000.0), ensemble.RandomForestClassifier(),

        linear_model.Perceptron(n_iter=40, eta0=0.1, random_state = 0)]

n_components_ = [7,21,35,37]

clf_accu = np.zeros((len(clfs), len(n_components_)))

print('ready to learn')
for i,n_component_ in enumerate(n_components_):

    print('----- # of components: %d -----' % (n_component_))

    pca_ = PCA(n_components=n_component_)

    pca_features = pca_.fit_transform(x_train)

    pca_test = pca_.transform(x_test)

    for j,clf_ in enumerate(clfs):        

        clf_.fit(pca_features,y_train)

        y_pred = clf_.predict(pca_test)

        #print('Accuracy with %d components: %0.3f' % (n_components_,metrics.accuracy_score(y_test,y_pred)))

        clf_accu[j,i] = metrics.accuracy_score(y_test,y_pred)
fig = plt.figure(figsize=(12,8))

colors = ['ro-','yo-','go-','bo-','ko-']

ax = fig.add_subplot(111)

legends = []

for i in range(len(clfs)):

    ax.plot(n_components_,clf_accu[i], colors[i])

    legends.append(str(type(clfs[i])).strip('>').split('.')[-1])

ax.set_title('Classifiers Performance over # of components')

ax.set_xlabel('Number of PCA components')

ax.set_ylabel('Accuracy')

ax.legend(legends)

plt.show()