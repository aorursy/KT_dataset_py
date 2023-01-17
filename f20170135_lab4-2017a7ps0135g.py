import numpy as np

import pandas as pd

import os
DATA_DIR = '/kaggle/input/eval-lab-4-f464/'

DATA_TRAIN = 'train.npy'

DATA_EVAL = 'test.npy'
data_train = np.load(os.path.join(DATA_DIR, DATA_TRAIN), allow_pickle = True)

data_eval = np.load(os.path.join(DATA_DIR, DATA_EVAL), allow_pickle = True)
print(data_train.shape)

print(data_train[:, 0][0])

print(data_train[:, 1][0].shape)



num_train = data_train.shape[0]

num_eval = data_eval.shape[0]
import matplotlib.pyplot as pl



def plot_ten_random(images, labels):

    num_images = images.shape[0]

    for i in range(10):

        pl.subplot(2, 5, i+1)

        j = (int)(np.random.randint(num_images))

        pl.imshow(images[j])

        pl.xlabel(labels[j])



    pl.show()



# for i in range(10):

#     pl.subplot(2, 5, i+1)

#     j = (int)(np.random.randint(num_train))

#     pl.imshow(data_train[j, 1])

#     pl.xlabel(data_train[j, 0])



# pl.show()
plot_ten_random(data_train[:, 1], data_train[:, 0])
print(np.unique(data_train[:, 0]))

print(np.unique(data_train[:, 0]).shape)
data_train[:, 1][0].shape



im_width = 50

im_height = 50
from skimage.color import rgb2gray



data_train_gray = np.ndarray((num_train, im_height, im_width))

labels = np.ndarray(num_train)



data_eval_gray = np.ndarray((num_eval, im_height, im_width))

l_eval = np.ndarray(num_eval)



for ix, data in enumerate(data_train):

#     print(ix)

    data_train_gray[ix, :, :] = rgb2gray(data[1])

    

for ix, data in enumerate(data_eval):

#     print(ix)

    data_eval_gray[ix, :, :] = rgb2gray(data[1])
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



labels = le.fit_transform(data_train[:, 0])
# for i in range(10):

#     pl.subplot(2, 5, i+1)

#     j = (int)(np.random.randint(num_train))

#     pl.imshow(data_train_gray[j, :, :], cmap = 'gray')

#     pl.xlabel(le.inverse_transform(labels)[j])



# pl.show()



plot_ten_random(data_train_gray, le.inverse_transform(labels))

plot_ten_random(data_eval_gray, np.zeros(num_eval))
from skimage import feature as fe



x_canny_0 = fe.canny(data_train_gray[0])



print(x_canny_0.shape)

pl.imshow(x_canny_0)

pl.show()
# x_daisy_0 = fe.daisy(data_train_gray[3])



# print(x_daisy_0.shape)



x_greycomatrix_0 = fe.greycomatrix((int)(data_train_gray[3]*255), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)



print(x_greycomatrix_0.shape)
# data_train_canny = np.zeros_like(data_train_gray)

# data_eval_canny = np.zeros_like(data_eval_gray)



data_train_daisy = np.ndarray((num_train, 5, 5, 200))

data_eval_daisy = np.ndarray((num_eval, 5, 5, 200))



# for i in range(num_train):

#     data_train_canny[i, :, :] = fe.canny(data_train_gray[i])

    

# for i in range(num_eval):

#     data_eval_canny[i, :, :] = fe.canny(data_eval_gray[i])

    

# print(data_train_canny.shape, data_eval_canny.shape)



for i in range(num_train):

    data_train_daisy[i, :, :, :] = fe.daisy(data_train_gray[i])

    

for i in range(num_eval):

    data_eval_daisy[i, :, :, :] = fe.daisy(data_eval_gray[i])

    

print(data_train_daisy.shape, data_eval_daisy.shape)
plot_ten_random(data_train_canny, le.inverse_transform(labels))

plot_ten_random(data_eval_canny, le.inverse_transform(labels))
# x = data_train_gray.reshape(num_train, im_width*im_height)

# x_eval = data_eval_gray.reshape(num_eval, im_width*im_height)



x = data_train_daisy.reshape(num_train, 5*5*200)

x_eval = data_eval_daisy.reshape(num_eval, 5*5*200)



# x = data_train_canny.reshape(num_train, im_width*im_height)

# x_eval = data_eval_canny.reshape(num_eval, im_width*im_height)
print(x.shape)

print(labels.shape)

print(x_eval.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.decomposition import PCA



n_comp = 200

# pca = PCA(n_components = n_comp, svd_solver = 'randomized', whiten = True).fit(x_train)

pca = PCA(n_components = n_comp, svd_solver = 'randomized', whiten = True).fit(x)
x_pca = pca.transform(x)

x_train_pca = pca.transform(x_train)

x_test_pca = pca.transform(x_test)

x_eval_pca = pca.transform(x_eval)
print(x_train_pca.shape)

print(x_eval_pca.shape)

print(x_pca.shape)
from sklearn.svm import SVC



s = SVC(gamma = 'scale')

# s.fit(x_train_pca, y_train)

s.fit(x_pca, labels)

# s.fit(x_train, y_train)
pred_test_label = le.inverse_transform(s.predict(x_test_pca))

# pred_test_label = le.inverse_transform(s.predict(x_test))

actual_test_label = le.inverse_transform(y_test)



print(pred_test_label[:10])



print(actual_test_label[:10])
np.sum(pred_test_label == actual_test_label)/pred_test_label.size
pred_eval_label = le.inverse_transform(s.predict(x_eval_pca))
pred_eval_label[:10]
df_out = pd.DataFrame(pred_eval_label, index = data_eval[:, 0], columns = ["Celebrity"])

df_out.index.name = 'ImageId'

df_out.head()
df_out.to_csv('sub11.csv')