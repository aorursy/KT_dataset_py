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
data = pd.read_csv('../input/global-wheat-detection/train.csv')
data.head()
bbox = data['bbox']
col0 = []

col1 = []

col2 = []

col3 = []



for i in range(len(bbox)):

    res_bb = bbox[i].strip('][').split(', ')

    for j in range(len(res_bb)):

        res_bb[j] = float(res_bb[j])

        

        if j == 0:

            col0.append(res_bb[j])

        elif j == 1:

            col1.append(res_bb[j])

        elif j == 2:

            col2.append(res_bb[j])

        elif j == 3:

            col3.append(res_bb[j])

        else:

            print("Hoooo")
print(col0[0],col3[0], col1[0], col2[0])
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

data['source'] = lb.fit_transform(data['source'])

source = data['source']

set(source)
data.drop(["bbox", 'source'], axis=1, inplace=True)

data['class'] = source

data['xmin'] = col0

data['ymin'] = col1

data['xmax'] = col2

data['ymax'] = col3
data = data.rename(columns={'image_id':'filename'})
data.head()
data.to_csv("./finaldata.csv", index=None)
d = pd.read_csv('./finaldata.csv')
d.head()