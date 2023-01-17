import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sample = pd.read_csv("../input/stage_1_sample_submission.csv")
train_labels  = pd.read_csv("../input/stage_1_train_labels.csv")
print("number of points in the 0 Neg class is %i" % (train_labels.drop_duplicates('patientId', keep = 'first'))[(train_labels.drop_duplicates('patientId', keep = 'first'))['Target'] == 0].shape[0])
print("number of points in the 1 pos class is %i" % (train_labels.drop_duplicates('patientId', keep = 'first'))[(train_labels.drop_duplicates('patientId', keep = 'first'))['Target'] == 1].shape[0]) 
print("The number of traning examples(data points) = %i " % train_labels.shape[0])
print("The number of features we have = %i " % train_labels.shape[1])
train_labels.describe()
train_labels.isna().sum()
train_labels
import pydicom

PathDicom = "../input/stage_1_train_images/"
images = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        images.append((os.path.join(dirName,filename),filename))
import matplotlib.pyplot as plt
import matplotlib.patches as patches

f, ax = plt.subplots(2, 2, figsize=(25,20))
image_index = 0
for i in ax:
    for j in i:
        print(j)
        data = pydicom.read_file(images[image_index][0])
        print("/////////////////////////////////////////////////\n", data, "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/n" )
        image = data.pixel_array
        j.imshow(image) 
        rows = train_labels[train_labels["patientId"].str.match(images[image_index][1][:-4])]
        print( rows )
        for index, row in rows.iterrows():
            rect = patches.Rectangle((row['x'],row['y']),row['width'],row['height'],linewidth=5,edgecolor='b',facecolor='none')
            j.add_patch(rect)
        image_index += 1
    