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
wines = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wines.head()
wines.head(3)
wines["country"]
wines[["country", "price"]]
wines
country_prices = wines[["country", "price"]]
wines["country"].unique()
wines["country"].value_counts()
wines["country"].value_counts().plot(kind="bar")
wines["price"].min()
wines["price"].max()
wines["price"].mean()
wines["price"].median()
maxprice = wines["price"].max()
maxprice
wines[wines["country"] == "Germany"]
wines[wines["price"] == 3300]
wines[wines["price"] == maxprice]