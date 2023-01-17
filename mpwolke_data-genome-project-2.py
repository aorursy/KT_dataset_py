#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQNrAdEBt3LIAnP-zl1vS-SsfP6FtTM8JdEu-KBuMo1D8PJB4RJ&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import cv2

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/buildingdatagenome1.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/NewWay.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/pred_longterm2.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/pred_longterm1.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/meters_month.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/meters_weekday.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/heatmap.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/dataQuality_all.jpg")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/map.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/weather_hour.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/meters_hist.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/meters_hour_usage.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/weather_corr.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/breakOut_all.jpg")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/Oldway.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/pred_shortterm-summer2.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/weatherSensitivity_all.jpg")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/split_shorterm.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/normalizedConsumption_all.jpg")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/weather_features.png")

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/building-data-genome-project-2/figures/buildingdatagenome2.png")

plt.imshow(im)

display(plt.show())