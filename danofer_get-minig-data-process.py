# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()
df.shape
df.head()
df2 = df.drop_duplicates(subset=["date","% Silica Concentrate","% Iron Concentrate"]).drop("% Iron Concentrate",axis=1)
df2.shape
df.to_csv("MiningProcessPlant_Database.csv.gz",index=False,compression="gzip")
df2.to_csv("MiningProcessPlant_lavels.csv.gz",index=False,compression="gzip")
