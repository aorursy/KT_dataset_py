import numpy as np
import pandas as pd
import cv2
import os

%matplotlib inline
import matplotlib.pyplot as plt
ROOT = '/kaggle/input/bee-vs-wasp/kaggle_bee_vs_wasp'
df = pd.read_csv(os.path.join(ROOT, 'labels.csv'))
from tqdm import tqdm
for idx in tqdm(df.index):
    df.loc[idx, 'path'] = df.loc[idx,'path'].replace('\\', '/')
counts = df['label'].value_counts()
labels = counts.index.tolist()
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Unique values of the original data')
plt.show()
labels = list(df['photo_quality'].unique())
x = range(0, 2)
y = list(df['photo_quality'].value_counts())
plt.bar(x, y, tick_label=labels)
plt.title('High quality photos in original data (1=high, 0=low)')

plt.show()
def img_plot(df, root, label):
    """show the first 9 images that match the given label"""
    df = df.query('label == @label')
    imgs = []
    for path in df['path'][:9]:
        img = cv2.imread(os.path.join(root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    f, ax = plt.subplots(3, 3, figsize=(15,15))
    for i, img in enumerate(imgs):
        ax[i//3, i%3].imshow(img)
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('label: %s' % label)
    plt.show()

# show the first 9 bees
img_plot(df, ROOT, 'bee')
def create_dataset(df, root, img_size):
    """Read dataset and convert images to img_size X img_size"""
    img_length = 3 * img_size * img_size
    imgs = []
    lbls = []
    for path, label in zip(tqdm(df['path']), df['label']):
        img = cv2.imread(os.path.join(root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size,img_size))
        imgs.append(np.array(img).reshape(img_length))
        lbls.append(label)
        
    imgs = np.array(imgs, dtype='float32')
    imgs = imgs / 255.0
    lbls = np.array(lbls)
    return imgs, lbls

X, y = create_dataset(df, ROOT, 128)
# Here we split the entire dataset into a training set (70%) and a test set (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
y_train_bee = y_train == 'bee'
y_test_bee = y_test == 'bee'
