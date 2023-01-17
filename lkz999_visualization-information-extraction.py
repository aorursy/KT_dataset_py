import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

import os

from skimage import exposure

import pydicom

import glob

from pydicom.data import get_testdata_files
train = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")

sample_sub = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")
train_path = ("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images")

test_path = ("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images")
print("Train shape : {}".format(train.shape))

print("Test shape : {}".format(sample_sub.shape))
train['ImageID'] = train['ID'].apply(lambda x: 'ID_' + x.split('_')[1] + '.dcm')

train['type'] = train['ID'].apply(lambda x: x.split('_')[2])

sample_sub ['ImageID'] =sample_sub['ID'].apply(lambda x: 'ID_' + x.split('_')[1] + '.dcm')

sample_sub['type'] = sample_sub['ID'].apply(lambda x: x.split('_')[2])
print('There are {} images in the train data set'.format(len(train.ImageID.unique())))

print('There are {} images in the test data set'.format(len(sample_sub.ImageID.unique())))
train.type.unique().tolist()
value_dict = train.Label.value_counts().to_dict()

print(train.Label.value_counts())
fig, ax = plt.subplots()

sns.barplot(x=list(value_dict.keys()), y=list(value_dict.values()), ax=ax)

ax.set_title("the number of labels")

ax.set_xlabel("class")

print('{:.1f} % of images have at least one type of Hemorrhage.'.format((value_dict[1]/value_dict[0])*100))
type_dict = train[train['Label'] == 1].type.value_counts().to_dict()

print(train[train['Label'] == 1].type.value_counts())
fig, ax = plt.subplots(figsize = (10, 6))

sns.barplot(x=list(type_dict.keys()), y=list(type_dict.values()), ax=ax)

ax.set_title("the number of different type of Hemorrhage")

ax.set_xlabel("type")

type_dict
images1_count = train[(train['Label'] == 1) & (train['type'] != 'any')].pivot_table(values = 'Label', index = ['ImageID'], aggfunc = 'sum').Label.value_counts().to_dict()

# exclude the type "any"

fig, ax = plt.subplots(figsize = (10, 6))

sns.barplot(x=list(images1_count.keys()), y=list(images1_count.values()), ax=ax)

ax.set_title("the number of images with different different class of Hemorrhage")

ax.set_xlabel("number of Hemorrhage")

for index,count in images1_count.items():

    print('There are {} images have {} Hemorrhage'.format(count, index))
filename = pydicom.read_file(os.path.join(train_path, "ID_000039fa0.dcm"))
filename
info_dict = {'SOP Instance UID': (0x08, 0x18), 

             'Modality' : (0x08, 0x60),

             'Patient ID' : (0x10, 0x20),

             'Study Instance UID' : (0x20, 0x0d),

             'Series Instance UID' : (0x20, 0x0e),

             'Study ID' : (0x20, 0x10),

             'Image Position (Patient)' : (0x20, 0x32),

             'Image Orientation (Patient)' : (0x20, 0x37),

             'Samples per Pixel' : (0x28, 0x02),

             'Photometric Interpretation': (0x28, 0x04),

             'Rows': (0x28, 0x10),

             'Columns': (0x28, 0x11),

             'Pixel Spacing': (0x28, 0x30),

             'Bits Allocated' : (0x28, 0x100),

             'Bits Stored' : (0x28, 0x101),

             'High Bit' : (0x28, 0x102),

             'Pixel Representation' : (0x28, 0x103),

             'Window Center' : (0x28, 0x1050),

             'Window Width' : (0x28, 0x1051),

             'Rescale Intercept' : (0x28, 0x1052),

             'Rescale Slope' : (0x28, 0x1053)}
pivot = train[train['type'] != 'any'].pivot_table(values = 'Label', index = ['ImageID'], aggfunc = 'sum').reset_index()
fig = plt.figure(figsize=(24, 16))

for j in range(4):

    for i, image in enumerate(pivot[pivot.Label == 1].iloc[j*4:(j+1)*4,].ImageID.tolist()):

        ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])

        img = np.array(pydicom.read_file(os.path.join(train_path, image)).pixel_array)

        img = exposure.equalize_hist(img)

        plt.imshow(img, cmap = plt.cm.bone)

        ax.set_title('One Hemorrhage ' + image)
fig = plt.figure(figsize=(24, 16))

for j in range(4):

    for i, image in enumerate(pivot[pivot.Label == 2].iloc[j*4:(j+1)*4,].ImageID.tolist()):

        ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])

        img = np.array(pydicom.read_file(os.path.join(train_path, image)).pixel_array)

        img = exposure.equalize_hist(img)

        plt.imshow(img, cmap = plt.cm.bone)

        ax.set_title('Two Hemorrhage ' + image)
fig = plt.figure(figsize=(24, 16))

for j in range(4):

    for i, image in enumerate(pivot[pivot.Label == 3].iloc[j*4:(j+1)*4,].ImageID.tolist()):

        ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])

        img = np.array(pydicom.read_file(os.path.join(train_path, image)).pixel_array)

        img = exposure.equalize_hist(img)

        plt.imshow(img, cmap = plt.cm.bone)

        ax.set_title('Three Hemorrhage ' + image)
fig = plt.figure(figsize=(24, 16))

for j in range(4):

    for i, image in enumerate(pivot[pivot.Label == 4].iloc[j*4:(j+1)*4,].ImageID.tolist()):

        ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])

        img = np.array(pydicom.read_file(os.path.join(train_path, image)).pixel_array)

        img = exposure.equalize_hist(img)

        plt.imshow(img, cmap = plt.cm.bone)

        ax.set_title('Four Hemorrhage ' + image)
fig = plt.figure(figsize=(24, 16))

for j in range(4):

    for i, image in enumerate(pivot[pivot.Label == 5].iloc[j*4:(j+1)*4,].ImageID.tolist()):

        ax = fig.add_subplot(4, 4, j * 4 + i + 1, xticks=[], yticks=[])

        img = np.array(pydicom.read_file(os.path.join(train_path, image)).pixel_array)

        img = exposure.equalize_hist(img)

        plt.imshow(img, cmap = plt.cm.bone)

        ax.set_title('Five Hemorrhage ' + image)
def get_info(data, prefix) : 

    for keys, values in info_dict.items():

        data[key] = data['ImageID'].apply(lambda x: (pydicom.read_file(os.path.join(prefix, x))[values].value))

    return data
train = get_info(train, train_path)

sample_sub = get_info(sample_sub, test_path)
train.to_pickle('train_info.pkl')

sample_sub.to_pickle('test_info.pkl')
# means = train.groupby('type').mean()

# sample_sub.loc[sample_sub['type'] == 'epidural', 'Label'] = means.Label[1]

# sample_sub.loc[sample_sub['type'] == 'intraparenchymal', 'Label'] = means.Label[2]

# sample_sub.loc[sample_sub['type'] == 'intraventricular', 'Label'] = means.Label[3]

# sample_sub.loc[sample_sub['type'] == 'subarachnoid', 'Label'] = means.Label[4]

# sample_sub.loc[sample_sub['type'] == 'subdural', 'Label'] = means.Label[5]

# sample_sub.loc[sample_sub['type'] == 'any', 'Label'] = means.Label[0]

# sample_sub[['ID', 'Label']].to_csv('submission.csv', index=False)