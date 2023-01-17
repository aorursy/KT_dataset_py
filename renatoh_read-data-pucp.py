import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/train_v2.csv')
print(df.shape)
df.head(3)
np.random.seed(34)
sample = 10_000
df = df.sample(sample)
df.shape
img_size = 96

def read_img(path):
    x = cv2.imread('../input/train-jpg/'+path+'.jpg')
    x = cv2.resize(x, (img_size, img_size))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x
from joblib import Parallel, delayed

with Parallel(n_jobs=12, prefer='threads', verbose=1) as ex:
    x = ex(delayed(read_img)(file) for file in df.image_name)
    
x = np.stack(x)
x.shape
labels = sorted({ee for e in df.tags.unique() for ee in e.split(' ')})
labels
for lbl in labels:
    df[lbl] = df.tags.str.contains(lbl)
    
df.head(3)
y =  df.iloc[:,2:].astype(np.int).values
y
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
def plot_img(x, y):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,6))
    lbls = [lbl for lbl,prob in zip(labels, y) if prob == 1]
    ax1.imshow(x)
    ax1.set_axis_off()
    ax1.set_title('\n'.join(lbls), size=14)
    ax2.bar(np.arange(len(y)), y)
    ax2.set_xticks(np.arange(len(y)))
    ax2.set_xticklabels(labels, rotation=90)
    plt.show()
idx = np.random.choice(len(x_train))
sample_x, sample_y = x_train[idx], y_train[idx]
plot_img(sample_x, sample_y)



