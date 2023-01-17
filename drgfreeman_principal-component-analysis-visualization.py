import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.decomposition import PCA
!pip install git+https://github.com/DrGFreeman/rps-cv.git
!ln -s ../input/rps-cv-images ./img
from rpscv import imgproc
X, y = imgproc.generateGrayFeatures()
X.shape, y.shape
img = X[100].reshape((200, 300))

plt.imshow(img, cmap='gray')

plt.xticks([])

plt.yticks([])

plt.show()
np.unique(y, return_counts=True)
for i in np.unique(y):

    print(i, imgproc.utils.gestureTxt[i])
pca = PCA(n_components=40)

pca.fit(X)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.scatter(range(pca.n_components_), pca.explained_variance_ratio_)

plt.xlabel('Principal component')

plt.ylabel('Explained variance ratio')

plt.title('Explained variance ratio by principal component')

plt.grid()

plt.subplot(1, 2, 2)

plt.plot(pca.explained_variance_ratio_.cumsum())

plt.xlabel('Principal component')

plt.ylabel('Explained variance ratio')

plt.title('Cummulative explained variance ratio')

plt.grid()

plt.tight_layout()

plt.show()
pca.components_.shape
plt.imshow(pca.components_[0].reshape((200, 300)), cmap='gray')

plt.xticks([])

plt.yticks([])

plt.show()
pca.mean_.shape
plt.imshow(pca.mean_.reshape((200, 300)), cmap='gray')

plt.xticks([])

plt.yticks([])

plt.show()
pc_imgs = pca.components_.reshape((len(pca.components_), 200, 300))



nb_col = 4

nb_row = pc_imgs.shape[0] // nb_col

plt.figure(figsize=(4 * nb_col, 3.2 * nb_row))

for i in range(nb_col * nb_row):

    plt.subplot(nb_row, nb_col, i+1)

    plt.imshow(pc_imgs[i], cmap='gray')

    plt.title("Principal component {:d}\nExpl. var. ratio {:.4f}".format(i, pca.explained_variance_ratio_[i]))

    plt.xticks(())

    plt.yticks(())

plt.show()
X_pca = pca.transform(X)



X_pca.shape
X_pca[123]
img_from_pcs = np.dot(X_pca[123], pca.components_) + pca.mean_



img_from_pcs.shape
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.imshow(X[123].reshape((200, 300)), cmap='gray')

plt.title('Original image')

plt.xticks([])

plt.yticks([])

plt.subplot(1, 2, 2)

plt.imshow(img_from_pcs.reshape((200, 300)), cmap='gray')

plt.title('Image from principal components')

plt.xticks([])

plt.yticks([])

plt.tight_layout()

plt.show()
X_from_pcs = pca.inverse_transform(X_pca)



X_from_pcs.shape
rock = X[100]

paper = X[800]

scissors = X[1500]
def show_img_pcs(img):

    plt.figure(figsize=(16, 4))

    

    # Display original image

    plt.subplot(1, 3, 1)

    plt.imshow(img.reshape(200, 300), cmap='gray');

    plt.title('Original image')

    plt.xticks(())

    plt.yticks(())

    

    #Display principal components magnitude

    plt.subplot(1, 3, 2)

    img_pc = pca.transform([img])

    plt.bar(range(1, img_pc.shape[1] + 1), img_pc[0,:])

    plt.title('Image principal components magnitude')

    plt.xlabel('Principal component')

    plt.ylabel('Magnitude')

    

    # Display reconstituted image

    plt.subplot(1, 3, 3)

    plt.imshow(pca.inverse_transform(img_pc).reshape(200, 300), cmap=plt.cm.gray)

    plt.title('Image reconstituted from principal components')

    plt.xticks(())

    plt.yticks(())

    

    plt.tight_layout()

    plt.show()
show_img_pcs(rock)
show_img_pcs(paper)
show_img_pcs(scissors)
def progressive_plot(index):

    img = X[index]

    img_pc = X_pca[index]

    plt.figure(figsize=(16, 4 * (pca.n_components_ + 1)))

    for i in range(-1, pca.n_components_):

        # Display the bargraph of image pc features

        plt.subplot(pca.n_components_ + 1, 3, 3 * i + 4)

        if i == -1:

            plt.imshow(img.reshape((200, 300)), cmap='gray')

            plt.title('Original image')

            plt.xticks([])

            plt.yticks([])

        else:

            bars = plt.bar(range(pca.n_components_), img_pc, color='lightgray')

            for j in range(i):

                bars[j].set_color('#6495ED')

            bars[i].set_color('r')

            plt.title('Image principal components magnitude')

            plt.xlabel('Principal component')

            plt.ylabel('Magnitude')

        # Display the left image (principal component vector being added)

        plt.subplot(pca.n_components_ + 1, 3, 3 * i + 5)

        if i == -1:

            plt.imshow(pca.mean_.reshape((200, 300)), cmap='gray')

            plt.title('Mean')

            plt.xticks([])

            plt.yticks([])

        else:

            plt.imshow((img_pc[i] * pca.components_[i]).reshape((200, 300)), cmap='gray')

            plt.title('Principal component vector {} * {:.3g}'.format(i, img_pc[i]))

            plt.xticks([])

            plt.yticks([])

        # Display the right image (progressively reconstituted image)

        plt.subplot(pca.n_components_ + 1, 3, 3 * i + 6)

        if i == -1:

            plt.imshow(pca.mean_.reshape((200, 300)), cmap='gray')

            plt.title('Mean')

            plt.xticks([])

            plt.yticks([])

        else:

            plt.imshow((np.dot(img_pc[:i+1], pca.components_[:i+1]) + pca.mean_).reshape((200, 300)), cmap='gray')

            plt.title('Principal components 0-{} + mean'.format(i))

            plt.xticks([])

            plt.yticks([])

    plt.tight_layout()

    plt.show()
progressive_plot(800)
progressive_plot(100)
progressive_plot(1500)