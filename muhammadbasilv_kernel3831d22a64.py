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
import pandas as pd

df=pd.read_csv('../input/lastone/sample_submission.csv')
df.head()
dict={}
for i in range(10):

    dict[df['image_id'][i]]=df['PredictionString'][i]
dict
df=pd.DataFrame.from_dict(dict,orient='index')
df.head()
df.index.names=['image_id']

df.columns=['PredictionString']
df.head()
test_list=['../input/global-wheat-detection/test/2fd875eaa.jpg','../input/global-wheat-detection/test/348a992bb.jpg','../input/global-wheat-detection/test/51b3e36ab.jpg','../input/global-wheat-detection/test/51f1be19e.jpg','../input/global-wheat-detection/test/53f253011.jpg','../input/global-wheat-detection/test/796707dd7.jpg','../input/global-wheat-detection/test/aac893a91.jpg','../input/global-wheat-detection/test/cb8d261a3.jpg','../input/global-wheat-detection/test/cc3532ff6.jpg','../input/global-wheat-detection/test/f5a1f0358.jpg']
import cv2

for i in test_list:

    cv2.imread(i)
df.to_csv('submission.csv')