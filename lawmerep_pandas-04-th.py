# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filePaths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        print(path)

        filePaths.append(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def viewInfo(table, name="Table"):

    col = table.index.tolist()

    row = table.columns.tolist()

    colLenght = len(col)

    rowLenght = len(row)

    print('[------{0}------]\n[*] Row : {1} | Col {2}'.format(name, rowLenght, colLenght))

    print(row, '\n')
# Hiển thị dữ liệu cho use_log.csv

uselog = pd.read_csv(filePaths[0])

viewInfo(uselog)

uselog.isnull().sum()
# Hiển thị dữ liệu cho customer_join.csv

customer = pd.read_csv(filePaths[1])

viewInfo(customer)

customer.isnull().sum()
# Đánh giá dữ liệu

customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]

viewInfo(customer_clustering)

customer_clustering.head()