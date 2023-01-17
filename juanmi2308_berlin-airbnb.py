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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np





df = pd.read_csv("../input/listings.csv")

df



df2 = df[df["neighbourhood"]=="Brunnenstr. Süd"]

df2



df3 = df2[df2["room_type"]=="Private room"]

df3



df4 = df3[df3["host_name"]=="Alexander"]

df4



g1 = df4.plot.bar(x="id", y="price", rot=0)
df5 = pd.DataFrame({"Price": df4["price"],

                   "Minimum amount of nights": df4["minimum_nights"]})



df5.plot.bar(rot=0)
df6 = pd.DataFrame({"Availability": df4["availability_365"],

                   "Minimum amount of nights": df4["minimum_nights"]})



df6.plot.bar(rot=0)