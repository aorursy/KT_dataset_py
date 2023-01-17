# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data_path = "../input/"

file_name = data_path + "Demonetization_data.csv"
df = pd.read_csv(file_name, encoding = "ISO-8859-1")

df.head()
df.shape