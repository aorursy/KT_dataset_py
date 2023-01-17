# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%pylab --no-import-all

%matplotlib inline

import numpy as np

import pandas as pd

import dicom

import os



images_path = '../input/sample_images/'
def get_3d_data(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    return np.stack([s.pixel_array for s in slices])
patients = os.listdir(images_path)



sample_image = get_3d_data(images_path + patients[0])

sample_image.shape
sample_image[sample_image==-2000]=0
#same plane as the original data, cut at the Z axis

pylab.imshow(sample_image[100], cmap=pylab.cm.bone)

pylab.show()
#remaping the image to 1 standard deviation of the average and clipping it to 0-1

img_std = np.std(sample_image)

img_avg = np.average(sample_image)

std_image = np.clip((sample_image - img_avg + img_std) / (img_std * 2), 0, 1)
#same cut as before, a bit easier to spot the features

pylab.imshow(std_image[100], cmap=pylab.cm.bone)

pylab.show()