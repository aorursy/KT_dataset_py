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
ss = pd.read_csv("../input/train_masks.csv")
ss.head()
def imagePrint(filePath):
    im = Image.open(filePath)
    image = np.array(im)
    io.imshow(image)
    io.show()
from skimage import data, io, filters
from PIL import Image
import time

import subprocess
ls = subprocess.Popen(["ls", "../input/train"], stdout = subprocess.PIPE,)
head = subprocess.Popen(["head"], stdin = ls.stdout, stdout = subprocess.PIPE,)

for fileName in head.stdout:
    filePath = "../input/train/" + fileName.decode("utf-8").replace("\n", "")
    print(filePath)
    imagePrint(filePath)
    time.sleep(15)

    
fileName
imagePrint("../input/train/10_10.tif")
"../input/train/" + fileName.decode("utf-8").replace("\n", "")
type(fileName)
fileName.decode("utf-8")
