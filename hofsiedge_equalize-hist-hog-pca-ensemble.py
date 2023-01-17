import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.utils import shuffle



%matplotlib inline
DATASET_ROOT = "../input"



X_train, y_train = shuffle(np.load(f'{DATASET_ROOT}/x_train.npy'), np.load(f'{DATASET_ROOT}/y_train.npy'))

X_test = np.load(f'{DATASET_ROOT}/x_test.npy')
size = 80

shape =  2 * (size,)

width, height = 8, 8



plt.figure(figsize=(16, 20))

for n, (image, name) in enumerate(zip(X_train, y_train), 1):

    if n > width * height:

        break

        

    plt.subplot(height, width, n)

    plt.title(name)

    plt.imshow(image.reshape(shape), cmap='gray')
from skimage import exposure

from skimage.feature import hog



from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()
img_shape = (80, 80)

def preprocess_images(*image_sets, concat=False):

    result = list(map(exposure.equalize_hist,

                      [img.reshape(img_shape) for img in np.concatenate(image_sets)]))

    result = [img.reshape((1, 6400)) for img in result]

    result = np.sqrt(result)

                  

    return np.concatenate(result) if concat else result
X_train_pp = preprocess_images(X_train)

X_test_pp  = preprocess_images(X_test)
img = X_train[170]



fd, hog_image = hog(image.reshape(img_shape), orientations=5, pixels_per_cell=(10, 10), 

                    visualise=True, cells_per_block=(3, 3), block_norm='L2-Hys')



plt.figure(figsize=(8,4))

plt.subplot(1,2,1)

plt.imshow(img.reshape((80,80)), cmap='gray')

plt.subplot(1,2,2)

plt.imshow(hog_image)

print(fd.shape)
img_shape = (80, 80)

orientations, p_p_c, c_p_b = 3, (8,) * 2, (3,) * 2  



def apply_hog(*image_sets):

    result = []

    for i in range(len(image_sets)):

        image_set = image_sets[i]

        result.append(

            np.array(

                [hog(img.reshape(img_shape), orientations=orientations, 

                     pixels_per_cell=p_p_c, cells_per_block=c_p_b, block_norm='L2-Hys') 

                 for img in image_set]))

        print(f'\r{i + 1}\t/\t{len(image_sets)}', end='')

    std_scaler.fit(np.concatenate(result))

    

    return list(map(std_scaler.transform, result))
train_hog_fts, test_hog_fts = apply_hog(X_train_pp, X_test_pp)
from sklearn.decomposition import PCA
pca = PCA(n_components=360)

pca.fit(np.concatenate([train_hog_fts, test_hog_fts]))
data_for_PCA_test = pca.transform(np.concatenate([train_hog_fts, test_hog_fts]))
explained_variance = pca.explained_variance_

explained_variance_ratio = pca.explained_variance_ratio_

plt.plot(np.cumsum(explained_variance))

print(np.sum(explained_variance), np.sum(explained_variance_ratio))
train_hog_pca = pca.transform(train_hog_fts)

test_hog_pca  = pca.transform(test_hog_fts)



std_scaler.fit(np.concatenate([train_hog_pca, test_hog_pca]))

train_hog_pca = std_scaler.transform(train_hog_pca)

test_hog_pca = std_scaler.transform(test_hog_pca)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
classifier = VotingClassifier(n_jobs=-1, voting='hard', weights=None, 

                              estimators=[

                                  ('nc', NearestCentroid(metric='cosine', shrink_threshold=0.15)), 

                                  ('rnc', RadiusNeighborsClassifier(metric='cosine', radius=0.9, weights='distance', outlier_label=1)),

                                  ('knn', KNeighborsClassifier(metric='cosine', n_neighbors=18, weights='distance'))

                              ]

                             )
cv_score = cross_val_score(estimator=classifier, X=train_hog_pca, cv=5, y=y_train, n_jobs=-1, verbose=0)

(cv_score, cv_score.mean())
classifier.fit(train_hog_pca, y_train)
import pandas as pd



pred = classifier.predict(test_hog_pca)



pd.DataFrame({'Id': np.arange(1, len(pred)+1), 'Name': pred}).to_csv('prediction_kaggle_kernel.csv', index=False)