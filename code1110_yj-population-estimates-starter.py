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
# February 2020

df2020_02 = pd.read_excel("/kaggle/input/the-number-of-visitors-estimate-in-tokyo-23-wards/230407.xlsx", sheet_name="2020.02", encoding="cp932")

print(df2020_02.shape)

df2020_02.head()
# March 2020

df2020_03 = pd.read_excel("/kaggle/input/the-number-of-visitors-estimate-in-tokyo-23-wards/230407.xlsx", sheet_name="2020.03", encoding="cp932")

print(df2020_03.shape)

df2020_03.head()
# April 2020

df2020_04 = pd.read_excel("/kaggle/input/the-number-of-visitors-estimate-in-tokyo-23-wards/230407.xlsx", sheet_name="2020.04", encoding="cp932")

print(df2020_04.shape)

df2020_04.head()