# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image079.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image014.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image028.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image165.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image020.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image157.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image076.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image081.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image015.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image147.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image123.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image010.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image183.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image190.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image151.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image029.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image077.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image074.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image163.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image086.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image122.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image084.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image002.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import cv2

img = cv2.imread('/kaggle/input/msl-m-rems-2-edr-v1.0/mslrem_0001/DOCUMENT/REMS_CALIB_PLAN_archivos/image004.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)