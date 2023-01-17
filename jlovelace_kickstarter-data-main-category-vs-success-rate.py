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
ds = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")

ds.info()
ds[["main_category", "state"]]

ds["state"].value_counts()
compProj = ds[(ds["state"] != "undefined") & (ds["state"] != "live")]

compProj["state"].value_counts()
totProjCount = compProj["main_category"].value_counts()

print(totProjCount)
susProj = compProj[compProj["state"] == "successful"]

susProjCount = susProj["main_category"].value_counts()

print(susProjCount)
percentSuccess = susProjCount/totProjCount * 100

percentSuccess.sort_values()