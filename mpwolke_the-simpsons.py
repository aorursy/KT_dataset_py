# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/050d45c5-5316-4c99-b523-f5fb986f7cd2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/00f102ca-7986-408b-a368-11f170735eb4.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/018fb8ab-f7cc-4592-ba87-ae916cdbfae9.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/010cff5a-36d8-4017-a6cd-0a3aba3f70a5.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/0252c242-9c4d-40ec-b0ce-539b47c37605.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/00ac437e-6afd-40b1-b33e-bbd73e6224ee.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/014a1f24-a9ac-408e-bedd-83abd38bc8cb.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/01fbad98-405b-4f9b-99d6-df8100914e3a.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/8878b106-cb6d-4627-a026-68f14f881534.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZMlkzwHMc0lLD23ODgvQdYhlZZK5gd2Vt6RKHP0nVbRoRjF4m&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQslKeJB9mitqrlUJoiLrzAxiUMf6V4TCSR3IBa_ke-F9-1cXq0OQ&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYXHE6FJO2GXs7JFAtScfU_etUpK61Zqz87IACnsNXP_uMKwnMUA&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVU2CDUFOyobV-VrbldvtKAV4Yu8trWzssfebf93Y4_eTISsxOcQ&s',width=400,height=400)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/the-simpsons-annotated/the-simpsons-dataset/e4d2e7c8-7d2f-47b7-bd95-dbc095a661cd.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzwATr0-JMyVwB09EIrGc_Y-9Pn-RICIS-ZrcZMQy8pIi-1Bs4&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRz_yJ_I54vauahnb9zBDe8Qb9Kf9Qj3gdBo9lDlpQtaVUdyRaIUw&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_HtEPOOkSQzMcCzQGGbWkCykWLNLYj78J35DqKvItgYASTEn_SQ&s',width=400,height=400)