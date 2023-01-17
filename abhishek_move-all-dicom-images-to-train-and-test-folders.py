import shutil

import os

import glob2
train_path = '../input/train/'

test_path = '../input/test/'
for filename in glob2.glob('../input/dicom-images-train/**/*.dcm'):

    fname = str(filename).split('/')[-1]

    print(fname)

    shutil.copy(str(filename), os.path.join(train_path, fname))
for filename in glob2.glob('../input/dicom-images-test/**/*.dcm'):

    fname = str(filename).split('/')[-1]

    print(fname)

    shutil.copy(str(filename), os.path.join(test_path, fname))