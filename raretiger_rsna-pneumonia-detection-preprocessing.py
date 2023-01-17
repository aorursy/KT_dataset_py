# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pydicom as pdcm
import pylab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
input_path = "../input/"
# Any results you write to the current directory are saved as output.
def load_dcm_data(filename):
    # attr = ["Rows", "Columns", "PixelSpacing"]
    dcm_data = pdcm.read_file(filename)
    dcmImg = dcm_data.pixel_array
    dcm_row = int(dcm_data.get("Rows"))
    dcm_colum = int(dcm_data.get("Columns"))
    dcm_spacing = dcm_data.get("PixelSpacing")
    dcm_spacing = [float(dcm_spacing[0]), float(dcm_spacing[1])]

    return dcmImg, [dcm_row, dcm_colum, dcm_spacing[0], dcm_spacing[1]]
train_labels_info = pd.read_csv(input_path + 'stage_1_train_labels.csv')
train_ID_list = train_labels_info['patientId'].tolist()
train_target = train_labels_info['Target'].tolist()
train_UniqID = train_labels_info['patientId'].unique().tolist()

train_mask_info = []
for x, y, w, h in zip(train_labels_info['x'].tolist(), train_labels_info['y'].tolist(), train_labels_info['width'].tolist(), train_labels_info['height'].tolist()):
    if np.isnan(x):
        x, y, w, h = 0, 0, 0, 0
    
    train_mask_info.append([int(x), int(y), int(w), int(h)])
# train_mask_info.extend(train_labels_info['x'].tolist())
# train_mask_info.extend(train_labels_info['y'].tolist())
# train_mask_info.extend(train_labels_info['width'].tolist())
# train_mask_info.extend(train_labels_info['height'].tolist())
print(train_mask_info[5])
t = 0
tempID = [] #train_ID_list[4] 
zeroMask = np.zeros([1024, 1024])
trainImage = np.zeros([1024, 1024])

# for ID, mInfo in zip(train_ID_list, train_mask_info): 
ID = train_ID_list[5] 
mInfo = train_mask_info[5]

if ID in tempID:
    t -= 1
    zeroMask[int(mInfo[1]):int(mInfo[1] + mInfo[3]),
            int(mInfo[0]):int(mInfo[0] + mInfo[2])] = 1

else:
    filename = "../input/stage_1_train_images/" + ID + '.dcm'
    train_img, train_info = load_dcm_data(filename)

    trainImage[:,:] = train_img.copy()
    zeroMask[int(mInfo[1]):int(mInfo[1] + mInfo[3]), 
             int(mInfo[0]):int(mInfo[0] + mInfo[2])] = 1


tempID = ID
t += 1

fig, ax = plt.subplots(1, 2)
ax[0].imshow(train_img)
ax[1].imshow(zeroMask)

plt.show()


