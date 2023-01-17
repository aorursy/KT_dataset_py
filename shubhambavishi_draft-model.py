# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

import cv2
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe("/kaggle/input/reuirements/colorization_deploy_v2.prototxt", "/kaggle/input/reuirements/colorization_release_v2.caffemodel")

pts = np.load("/kaggle/input/reuirements/pts_in_hull.npy")
class8 = net.getLayerId("class8_ab")

conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]

net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

image = cv2.imread("/kaggle/input/bw-image/b-wimg.jpg")

scaled = image.astype("float32") / 255.0

lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))

L = cv2.split(resized)[0]

L -= 50

net.setInput(cv2.dnn.blobFromImage(L))

ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]

colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)



colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")



import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(colorized)

cv2.imwrite('/kaggle/working/out.jpeg', colorized)
