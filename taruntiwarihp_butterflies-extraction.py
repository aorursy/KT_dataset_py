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
data = pd.read_csv('/kaggle/input/butterflies/multimedia.csv')
data.head()
from PIL import Image
import requests
from io import BytesIO
img = []
# extraxt all images from url 
for r in range(len(data)):
    url = data['identifier'][r]
    response = requests.get(url)
    img.append(Image.open(BytesIO(response.content)))
    print(r,'Image Processing done...')
img
data['Images'] = img
data.head()
data.to_csv("Butterfliess.csv",index=False)
# Click on BlueButton on top-right, copy and start coding, Happy kaggling! ;)