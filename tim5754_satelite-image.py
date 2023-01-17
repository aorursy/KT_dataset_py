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
import cv2

import matplotlib.pyplot as plt
!ls ../input/train_sm/
img = cv2.imread('../input/train_sm/set107_1.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('../input/train_sm/set107_2.jpeg')

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
plt.imshow(img2)
img.shape
df_sub = pd.read_csv("../input/sample_submission.csv")

df_sub
df_train = pd.read_csv("../input/train_sm/")
