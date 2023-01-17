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
# load csv file into kaggle notebook, store it in a variable
dataframe1 = pd.read_csv("/kaggle/input/covid19-us-county-jhu-data-demographics/covid_us_county.csv")

#
dataframe1.head(1000000)
dataframe1.describe()
import matplotlib.pyplot as plt
dataframe1["state"].value_counts()
dataframe1.boxplot(column="cases", by="state", figsize=(100,10)) 
dataframe1["county"].value_counts()
dataframe1.apply(lambda x: sum(x.isnull()), axis=0)
fig = plt.figure(figsize=(8,4))
axl = fig.add_subplot(121)
axl.set_xlabel("date")
axl.set_ylabel("cases")
axl.set_title("Total cases")
dataframe1.value_counts().plot(kind="bar")
