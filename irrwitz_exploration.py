import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import os
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
train_len = len(os.listdir("../input/train"))
test_len = len(os.listdir("../input/test"))
print("Train size is {}, test size is {}".format(train_len, test_len))
train_df = pd.read_csv("../input/train_labels.csv")
train_df.head()
negative_cases_train = train_df[train_df["label"] == 0]
postive_cases_train = train_df[train_df["label"] == 1]
print("Postive cases {:,}, negative cases {:,} in training set".format(len(postive_cases_train), len(negative_cases_train)))
# now lets write a helper function to show some images
def show(df):
    fig, ax = plt.subplots(2,5, figsize=(20,5))
    for i, row in enumerate(df.itertuples()):
        path = os.path.join('../input/train/', row.id)
        img = Image.open(path+'.tif')
        w,h = img.size
        cropped = img.crop((w//2 - 32//2, h//2 - 32//2, w//2 + 32//2, h//2 + 32//2))
        box = patches.Rectangle((32,32),32,32,linewidth=2,edgecolor='r', facecolor='none')
        ax[0,i].imshow(img)
        ax[0,i].add_patch(box)
        ax[0,i].set_title("Label: {}".format(row.label))
        ax[1,i].imshow(cropped)
        ax[0,0].set_ylabel('Sample', size='large')
        ax[1,0].set_ylabel('Cropped', size='large')
show(negative_cases_train[0:5])
show(postive_cases_train[0:5])
def load(row):
    path = os.path.join('../input/train/', row.id)
    img = Image.open(path+'.tif')
    a = np.array(img)
    row['image'] = a
    row['image_flattened'] = a.flatten()
    return row
def distribution_plot(sample_size=100):
    neg_sample = train_df.loc[train_df.label == 0][0:sample_size]
    pos_sample = train_df.loc[train_df.label == 1][0:sample_size]
    neg_sample = neg_sample.apply(load, axis=1)
    pos_sample = pos_sample.apply(load, axis=1)
    print("Positive samples size {}".format(len(pos_sample)))
    print("Negative samples size {}".format(len(neg_sample)))
    neg = neg_sample.image_flattened.values.tolist()
    pos = pos_sample.image_flattened.values.tolist()
    plt.figure("Distribution")
    sns.distplot(np.concatenate(neg), label='Negative')
    sns.distplot(np.concatenate(pos), label='Positive')
    plt.legend()
    plt.show()
distribution_plot()
distribution_plot(1000)
