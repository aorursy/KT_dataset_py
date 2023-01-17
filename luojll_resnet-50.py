%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing import image
from keras.applications import resnet50
def load_img(img_id, train_or_test, size=(224, 224)):
    IMG_DIR = '/kaggle/input/dog-breed-identification/'
    img = image.load_img(os.path.join(IMG_DIR, train_or_test, f'{img_id}.jpg'),target_size=size)
    return img

def predict(img, model):
    img = image.img_to_array(img)
    X = np.expand_dims(img, axis=0)
    X = resnet50.preprocess_input(X)
    r = model.predict(X)
    return resnet50.decode_predictions(r, top=1)
data_dir = '/kaggle/input/dog-breed-identification/'
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
model = resnet50.ResNet50()
fig = plt.figure(figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.05)
for i, row in labels.sample(n=6).reset_index().iterrows():
    ax = grid[i]
    img = load_img(row['id'], 'train')
    _, pred_breed, likelihood = predict(img, model)[0][0]
    img = image.img_to_array(img)
    ax.imshow(img / 255.)
    ax.text(10, 180, f'ResNet50: {pred_breed} ({likelihood:.2f})', color='w', backgroundcolor='k', alpha=0.8)
    ax.text(10, 200, f'Label: {row["breed"]}', color='k', backgroundcolor='w', alpha=0.8)
    ax.axis('off')
plt.show()