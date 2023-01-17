# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
%matplotlib inline
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from pickle import dump
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TEST1_DIR = 'test1'
TRAIN_DIR = 'train'
INPUT_DIR = '../input/dogs-vs-cats'
OUTPUT_DIR = '/kaggle/working/'
from zipfile import ZipFile
files_to_extract = [TEST1_DIR,TRAIN_DIR]

for Dataset in files_to_extract:
    with ZipFile('{}/{}.zip'.format(INPUT_DIR, Dataset),'r') as z:
        z.extractall('.')
from subprocess import check_output
print(check_output(['ls', '.']).decode('utf8'))
filenames = os.listdir(TRAIN_DIR)
filenames[:5]
labels = [1 if filename.startswith('dog') else 0 for filename in filenames]

labels[:5]
df = pd.DataFrame({
    'filename': filenames,
    'category': labels
})

df.head()
ax = df.category.value_counts().plot.bar(color=['dodgerblue', 'slategray'])
plt.title('Dogs and Cats images count')
plt.xlabel('Dog = 1 - Cat = 0')
plt.ylabel('samples count')
ax.set_xticklabels(['Dog', 'Cat'], rotation=0, fontsize=11)
plt.show()
random_filename = '{}/{}'.format(TRAIN_DIR, random.choice(filenames))
random_image = mpimg.imread(random_filename)
# random_image
plt.imshow(random_image)
plt.show()
def load_images_from_dir(dir_location, filenames):
    return [np.array(Image.open('{}/{}'.format(dir_location, filename)).resize((64, 64))) for filename in filenames]
    
images = np.array(load_images_from_dir(TRAIN_DIR, filenames))
images.shape
h, w, d = images[0].shape
images_resized = np.array([np.reshape(img, (w*h*d)) for img in images])
images_resized.shape
mlp = MLPClassifier()
mlp.fit(images_resized, labels)
mlp.score(images_resized, labels)
tree = DecisionTreeClassifier()
tree.fit(images_resized, labels)
tree.score(images_resized, labels)
plot_tree(tree)
forest = RandomForestClassifier()
forest.fit(images_resized, labels)
forest.score(images_resized, labels)
test_filenames = os.listdir(TEST1_DIR)
test_filenames[:5]
random_filename = '{}/{}'.format(TEST1_DIR, random.choice(test_filenames))
random_image = mpimg.imread(random_filename)
random_image.shape
plt.imshow(random_image)
plt.show()
random_image_resized = np.array(Image.fromarray(random_image).resize((64, 64)))
random_image_resized.shape
random_image_resized = np.reshape(random_image_resized, (w*h*d))
mlp_prediction = mlp.predict([random_image_resized])
mlp_prediction
tree_prediction = tree.predict([random_image_resized])
tree_prediction
forest_prediction = forest.predict([random_image_resized])
forest_prediction
test_images = np.array(load_images_from_dir(TEST1_DIR, test_filenames))
test_images.shape
test_images_resized = np.array([np.reshape(img, (w*h*d)) for img in test_images])
test_images_resized.shape
mlp_predictions = mlp.predict(test_images_resized)
mlp_predictions[:5]
tree_predictions = tree.predict(test_images_resized)
tree_predictions[:5]
forest_predictions = forest.predict(test_images_resized)
forest_predictions[:5]
df_test_mlp = pd.DataFrame({
    'filename': test_filenames,
    'category': mlp_predictions
})

df_test_mlp.head()
df_test_tree = pd.DataFrame({
    'filename': test_filenames,
    'category': tree_predictions
})

df_test_tree.head()
df_test_forest = pd.DataFrame({
    'filename': test_filenames,
    'category': forest_predictions
})

df_test_forest.head()
def autolabel(ax):
    """
    Attach a text label above each bar displaying its height
    """
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
ax = df_test_mlp.category.value_counts().plot.bar(color=['dodgerblue', 'slategray'])
plt.title('Dogs and Cats images count MLP')
plt.xlabel('Dog = 1 - Cat = 0')
plt.ylabel('samples count')
ax.set_xticklabels(['Dog', 'Cat'], rotation=0, fontsize=11)
autolabel(ax)
plt.show()
ax = df_test_tree.category.value_counts().plot.bar(color=['dodgerblue', 'slategray'])
plt.title('Dogs and Cats images count Desicion Tree')
plt.xlabel('Dog = 1 - Cat = 0')
plt.ylabel('samples count')
ax.set_xticklabels(['Dog', 'Cat'], rotation=0, fontsize=11)
autolabel(ax)
plt.show()
ax = df_test_forest.category.value_counts().plot.bar(color=['dodgerblue', 'slategray'])
plt.title('Dogs and Cats images count Random Forest')
plt.xlabel('Dog = 1 - Cat = 0')
plt.ylabel('samples count')
ax.set_xticklabels(['Dog', 'Cat'], rotation=0, fontsize=11)
autolabel(ax)
plt.show()
IMAGE_SIZE = (w, h)
sample_test = df_test_mlp.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = mpimg.imread('./test1/{}'.format(filename))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel('{} ({})'.format(filename, category))
plt.tight_layout()
plt.show()
sample_test = df_test_tree.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = mpimg.imread('./test1/{}'.format(filename))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel('{} ({})'.format(filename, category))
plt.tight_layout()
plt.show()
sample_test = df_test_forest.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = mpimg.imread('./test1/{}'.format(filename))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel('{} ({})'.format(filename, category))
plt.tight_layout()
plt.show()
os.chdir(OUTPUT_DIR)
FILE_NAME = 'forest_model.sav'
dump(forest, open(FILE_NAME, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)
# from IPython.display import FileLink
# FileLink(FILE_NAME)
submission_df = df_test_forest.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
