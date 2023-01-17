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
electricity = pd.read_csv("/kaggle/input/dutch-energy/Electricity/stedin_electricity_2017.csv")

electricity.head(5)
electricity.shape
ele_grp = electricity.groupby("purchase_area")

supply_count = ele_grp.purchase_area.size().sort_values(ascending = False)

supply_count
supply_count.plot(kind = "pie")
ele_grp.annual_consume.sum().sort_values(ascending = False)
each_point_mean = ele_grp.annual_consume.mean().round(2)

each_point_std = ele_grp.annual_consume.std().round(2)

each_point_max = ele_grp.annual_consume.max()

each_point_min = ele_grp.annual_consume.min()

each_point_median = ele_grp.annual_consume.median()

frame = {"average_consume": each_point_mean , "std": each_point_std , "max_consume": each_point_max , "min_consume": each_point_min

        ,"median_consume": each_point_median}

table = pd.DataFrame(frame).reset_index()

table.sort_values("average_consume" , ascending = False)

#each_point_std
each_point_median = ele_grp.annual_consume.median()

each_point_median
sw_ele = electricity[electricity.purchase_area == "Stedin Weert"]

sw_ele.head(10)
sw_ele2 = sw_ele.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]]

sw_ele2.head(10)
sw_ele2.mean()
average = sw_ele.annual_consume.mean()

sw_ele_low = sw_ele[sw_ele.annual_consume < average]

sw_ele_low.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]].mean().round(2)
average = sw_ele.annual_consume.mean()

sw_ele_lar = sw_ele[sw_ele.annual_consume > average]

sw_ele_lar.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]].mean().round(2)