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
import urllib
import json
import urllib.request
import sys

from PIL import Image
import requests
from io import BytesIO

import cv2

import shutil


if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlopen







with open("../input/output.json") as json_data:
      d = json.load(json_data)
      #obj= d["images"]
        
      for k in range(50):
        url = d["images"][k]["darwin_url"]
        file_name = d["images"][k]["file_name"]
        print(url)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        response = requests.get(url, stream=True)
        with open(file_name, 'wb') as out_file:
                  shutil.copyfileobj(response.raw, out_file)
        del response