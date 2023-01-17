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
data_a = {"Customer_ID":pd.Series([1,2,3,4,5,6]), "Product":pd.Series(["PC", "PC", "PC", "Monitor", "Keyboard", "Mouse"])}

df_a = pd.DataFrame(data_a)



data_b = {"Customer_ID": pd.Series([2,4,6,99]), "City":pd.Series(["Vienna","Krems", "Vienna", "Bregenz"])}

df_b = pd.DataFrame(data_b)

df_a.head(10)

df_b.head(10)
pd.merge(df_a, df_b, on="Customer_ID", how="inner")
pd.merge(df_a, df_b, on="Customer_ID", how="outer")
pd.merge(df_a, df_b, on="Customer_ID", how="left")
pd.merge(df_a, df_b, on="Customer_ID", how="right")