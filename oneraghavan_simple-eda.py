import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import pydicom

import matplotlib.pylab as plt

from matplotlib import rcParams

import os



rcParams['figure.figsize'] = 11.7,8.27



traindir = "/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images"

testdir = "/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images"
!ls /kaggle/input/rsna-intracranial-hemorrhage-detection/
train_csv = pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")
train_csv.head(5)
train_csv["type"] = train_csv["ID"].apply(lambda a:a.split("_")[2])

train_csv["ID"] = train_csv["ID"].apply(lambda a:"_".join(a.split("_")[0:2]))
train_csv.head(10)
sns.countplot(train_csv.Label)
train_csv.groupby("type").sum()




label_distribution = train_csv.groupby("type").sum().reset_index()

sns.barplot(x=label_distribution.type,y=label_distribution.Label)
type_count_distribution = train_csv.groupby("ID").sum().reset_index()

vc = type_count_distribution.Label.value_counts()

sns.barplot(x=vc.index,y=vc)
def plot_images(hem_type):

    images = train_csv[(train_csv["type"] == hem_type) & (train_csv["Label"] == 1)]["ID"].values[:100]

    width = 5

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,5))



    for im in range(0, height * width):

        image = pydicom.read_file(os.path.join(traindir,images[im]+ '.dcm')).pixel_array

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')



    plt.suptitle(hem_type)

    plt.show()
plot_images("intraparenchymal")
plot_images("epidural")
plot_images("intraventricular")
plot_images("subarachnoid")
plot_images("subdural")
train_csv.drop_duplicates(inplace=True)

pivot = train_csv.pivot(index='ID', columns='type', values='Label').reset_index()
corrs = pivot[["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]].corr()

corrs.style.background_gradient(cmap='coolwarm')