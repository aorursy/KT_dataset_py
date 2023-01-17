from IPython.display import YouTubeVideo
from IPython import display
video_id = 'oKzNUGz21JM'
YouTubeVideo(video_id, width = 800, height = 500)
# Imports for data loading and array math
import numpy as np
import pandas as pd

# Imports for
from ipywidgets import interact

# Imports for visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

# Imports for dimensionality reduction
from sklearn.manifold import MDS, TSNE

# Import for working with directories and files
import os

# Import for working with vision applications of fastai
from fastai.vision import *

# Import for data split
from sklearn.model_selection import train_test_split
sample_submission_path = '../input/digit-recognizer/sample_submission.csv'
train_path = '../input/digit-recognizer/train.csv'
test_path = '../input/digit-recognizer/test.csv'
train = pd.read_csv(train_path)

X = train.iloc[:, 1:].values
X = X.reshape((X.shape[0], 28, 28))/255.

Y = train.iloc[:, 0].values

X_test = pd.read_csv(test_path)
X_test = X_test.values.reshape((X_test.shape[0], 28, 28))/255.
@interact
def plot_train_set(target = range(10), batch = (0, 263, 1)):
    mpl.rcParams['figure.figsize'] = 6, 6
    class_pictures = X[Y == target]
    side = 3
    for i in range(side):
        for j in range(1, side + 1):
            plt.subplot(side, side, i * side + j)
            temp_index = batch * (side ** 2) + i * side + j - 1
            if temp_index < class_pictures.shape[0]:
                plt.imshow(class_pictures[temp_index], cmap = 'gray')
plot_train_set(target = 4, batch = 13)
plot_train_set(target = 5, batch = 15)
plot_train_set(target = 6, batch = 13)
plot_train_set(target = 9, batch = 391)
@interact
def plot_test_set(batch = (0, 3100, 1)):
    mpl.rcParams['figure.figsize'] = 6, 6
    side = 3
    for i in range(side):
        for j in range(1, side + 1):
            plt.subplot(side, side, i * side + j)
            temp_index = batch * (side ** 2) + i * side + j - 1
            if temp_index < X_test.shape[0]:
                plt.imshow(X_test[temp_index], cmap = 'gray')
plot_test_set(batch = 13)
count = 1000
embedding = MDS(n_components = 2, metric = True, random_state = 42)
X_train_transformed = embedding.fit_transform(X[:count].reshape((count, 28 * 28)))
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = 12, 10
plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10))
plt.title('linear MDS transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));
non_linear_embedding = MDS(n_components = 2, metric = False, random_state = 42)
X_train_non_linear = embedding.fit_transform(X[:count].reshape((count, 28 * 28)))
plt.scatter(X_train_non_linear[:, 0], X_train_non_linear[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10))
plt.title('non-linear MDS transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));
tsne_embedding = TSNE(n_components = 2, random_state = 42)
X_train_tsne_transformed = tsne_embedding.fit_transform(X[:count].reshape((count, 28 * 28)))
plt.scatter(X_train_tsne_transformed[:, 0], X_train_tsne_transformed[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10), s = 50)
plt.title('t-SNE transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));
new_path = '../working/'
new_train_path = new_path + 'train/'
new_valid_path = new_path + 'valid/'
new_test_path = new_path + 'test/'
os.mkdir(new_train_path)
os.mkdir(new_valid_path)
os.mkdir(new_test_path)
for target_label in [(str(i) + '/') for i in range(10)]:
    os.mkdir(new_train_path + target_label)
    os.mkdir(new_valid_path + target_label)
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.1, shuffle = True, random_state = 42, stratify = Y)
def save_image_data(X_data, path):
    for i in range(X_data.shape[0]):
        save_path = path + str(i) + '.png'
        mpl.image.imsave(save_path, X_data[i])
for label in range(10):
    X_data = X_train[y_train == label]
    save_path = new_train_path + str(label) + '/'
    save_image_data(X_data, save_path)
    
    X_data = X_valid[y_valid == label]
    save_path = new_valid_path + str(label) + '/'
    save_image_data(X_data, save_path)
    
save_image_data(X_test, new_test_path)
data = ImageDataBunch.from_folder(new_path, test = 'test')
data.show_batch(rows = 4, figsize = (8, 8))
learner = cnn_learner(data, models.resnet18, metrics = accuracy)
learner.fit_one_cycle(5, 1e-3)
learner.save('five_epochs')
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(12, figsize = (12, 12))
interp.plot_confusion_matrix(figsize = (12, 12), dpi = 70)
learner.unfreeze()
learner.fit_one_cycle(5, max_lr = slice(1e-5, 1e-4))
learner.save('unfreeze_5')