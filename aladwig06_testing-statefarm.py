# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
imglist = pd.read_csv("../input/driver_imgs_list.csv")
images = []

for i in range(0, 3):
    img = Image.open("../input/train/" + imglist.values[i][1] + "/" + imglist.values[i][2])
    img = np.mean(np.array(img.getdata()), axis=1)
    images.append(img)
images = np.array(images)
plt.imshow(np.reshape(images[2], [480,640]))
