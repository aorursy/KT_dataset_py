# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_ship_segmentations_v2.csv")
dfWtShipOnly = df.drop( df.index[df.EncodedPixels.apply(lambda x: not isinstance(x, str)).tolist()])
# Use rle and its position in the image as identifier of the image
dfWtShipOnly["rleAndPosition"] = dfWtShipOnly.EncodedPixels.apply(lambda x: ' '.join(x.split(" ")[1::2]) + ' ' + ' '.join([ str(int(hor) % 256) for hor in x.split(" ")[0::2]]) if (isinstance(x, str)) else x)
dfWtShipOnly.head()
#List in a new column all the ImageId where the 'rleAndPosition' occurs.
dfWtShipOnly["allSameRle"] = dfWtShipOnly["rleAndPosition"].apply(lambda x: dfWtShipOnly.ImageId[dfWtShipOnly["rleAndPosition"] == x].tolist())
dfWtShipOnly.head(10)
# Verify that 'rleAndPosition' values only occurs mainly a few times 
dfWtShipOnly["rleAndPosition"].value_counts().describe()
# Plot "allSameRle"

import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from PIL import Image

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = "../input/train_v2/" + image_id
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

maxRepetition = 8
plt.figure(figsize=(20,20))
ImgNumber = 10
i = 0
for y in range(ImgNumber) :
    image_id = dfWtShipOnly["ImageId"].tolist()[np.random.randint(0,len(dfWtShipOnly))]
    img = get_image_data(image_id, 'Train')
    ID_list = dfWtShipOnly[dfWtShipOnly["ImageId"] == image_id]["allSameRle"].tolist()[0]
    if len(ID_list) > maxRepetition: ID_list = ID_list[0:maxRepetition-1]
    for ID in ID_list :
        i += 1
        img = get_image_data(ID, 'Train')
        plt.subplot(ImgNumber,maxRepetition,i)
        plt.imshow(img, cmap='binary')
    i = maxRepetition * (y+1)
# Group the 'rleAndPosition' by ImageId  
dfWtShipOnlyUnique = dfWtShipOnly.groupby('ImageId')['allSameRle'].apply(lambda x: set(x.sum())) 
print(len(df))
print(len(dfWtShipOnly))
print(len(dfWtShipOnlyUnique))
print(len(dfWtShipOnlyUnique))
alreadyDropped = []
dfWtShipOnlyUniqueCopy = dfWtShipOnlyUnique
for itemKeeped in dfWtShipOnlyUnique.iteritems() :
    if not itemKeeped[0] in alreadyDropped :
        for itemToCheck in dfWtShipOnlyUnique.iteritems() :
            if itemToCheck[0] in itemKeeped[1] and not itemToCheck[0] in alreadyDropped and itemToCheck[0] != itemKeeped[0]:
                dfWtShipOnlyUnique = dfWtShipOnlyUnique.drop(itemToCheck[0])  
                alreadyDropped = alreadyDropped + [itemToCheck[0]]
print(len(dfWtShipOnlyUnique))
# Splitting
trainDfWtShipOnlyUnique=dfWtShipOnlyUnique.sample(frac=0.9,random_state=768)
validationDfWtShipOnlyUnique=dfWtShipOnlyUnique.drop(trainDfWtShipOnlyUnique.index)
# Save the labels
allUniqLabels = dfWtShipOnly.loc[[True if ID in dfWtShipOnlyUnique.index else False for ID in dfWtShipOnly["ImageId"]]]
allUniqLabels.to_csv('/kaggle/working/uniqueAllLabels.csv', index=True)
print(len(allUniqLabels))
trainUniqLabels = dfWtShipOnly.loc[[True if ID in trainDfWtShipOnlyUnique.index else False for ID in dfWtShipOnly["ImageId"]]]
trainUniqLabels.to_csv('/kaggle/working/uniqueTrainLabels.csv', index=True)
print(len(trainUniqLabels))
validationUniqLabels = dfWtShipOnly.loc[[True if ID in validationDfWtShipOnlyUnique.index else False for ID in dfWtShipOnly["ImageId"]]]
validationUniqLabels.to_csv('/kaggle/working/uniqueValidationLabels.csv', index=True)
print(len(validationUniqLabels))