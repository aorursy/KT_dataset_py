import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline
! ls ../input/
train_org = pd.read_csv("../input/train.csv")
print("train shape:", train_org.shape)
train_org.head(5)
pd.DataFrame(train_org.Id.value_counts().describe([0.25, 0.40, 0.75, 0.8, 0.9, 0.95, 0.99, 0.999])).T
_id_counts = train_org.Id.value_counts()
_id_counts[_id_counts <= 20].hist(bins=21)
_ = plt.xlim(0, 20)
import os

from PIL import Image
path_train = "../input/train/"
filenames = os.listdir(path_train)
filenames[0]
with Image.open(os.path.join(path_train, filenames[0]), 'r') as img:
    print("filename:", img.filename)
    print("image shape:", np.array(img).shape)
    plt.imshow(img)
with Image.open(os.path.join(path_train, "00d641885.jpg"), 'r') as img:
    print("filename:", img.filename)
    print("image shape:", np.array(img).shape)
    plt.imshow(img)
submission = pd.read_csv("../input/sample_submission.csv")
print("test shape:", submission.shape)
submission.head()
