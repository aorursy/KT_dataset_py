import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import os

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
rootDir="/kaggle/input/tweet-sentiment-extraction/"
dfTrain=pd.read_csv(os.path.join(rootDir,"train.csv"))
dfTest=pd.read_csv(os.path.join(rootDir,"test.csv"))
dfTrain.head()
dfTest.head()
dfTrain.sentiment.value_counts()
dfTest.sentiment.value_counts()
