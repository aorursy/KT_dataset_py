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
# data load

df = pd.read_csv("/kaggle/input/kakaocommerce-kpi-prediction/orders_sample_v0.2.csv")

df.head(10)
# EDA
# model
# print res

print("20200401,1000\n20200402,1000\n20200403,1000\n20200404,1000\n20200405,1000\n20200406,1000\n20200407,1000\n")