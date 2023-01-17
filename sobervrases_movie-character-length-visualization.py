# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
df=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
df.head()
df.columns
titles= list(df.original_title)
#type(titles)
titles[:5]
Frequency={}
for title in titles:
    length=len(title)
    if Frequency.get(length) is None:
        Frequency[length]=1
    else:
        Frequency[length]+=1
Frequency