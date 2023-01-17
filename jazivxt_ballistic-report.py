import numpy as np

import pandas as pd

import glob

from PIL import Image

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv') #ImageId_ClassId, EncodedPixels

train['class'] = train['ImageId_ClassId'].map(lambda x: x.split('_')[1])

train['path'] = train['ImageId_ClassId'].map(lambda x: '../input/train_images/' + x.split('_')[0])

train.head()
test = pd.DataFrame(glob.glob('../input/test_images/**'), columns=['path'])

test['ImageId'] = test['path'].map(lambda x: x.split('/')[-1])

test.head()
plt.imshow(Image.open(train.path[0]))
#Four Models Applied Here

sub = []

for i in range(4):

    #train here

    trainTemp = train[train['class'] == i+1].reset_index(drop=True).copy()

    #model = make_model.train(trainTemp)

    

    #predict here

    subTemp = test.copy()

    subTemp['ImageId_ClassId'] = subTemp.apply(lambda r: '_'.join([r['ImageId'], str(i+1)]), axis=1)

    #subTemp['EncodedPixels'] = model.predict(subTemp)

    subTemp['EncodedPixels'] = None

    sub.append(subTemp.copy())

sub = pd.concat(sub)

sub[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)