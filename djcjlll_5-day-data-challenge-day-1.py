# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#print the first five lines and each col data class

with open('../input/train.csv') as f:

    f_csv = pd.read_csv(f)

    #print(f_csv.head())

print(f_csv.dtypes)
# statistic the train.csv contain how many days data, how many stores and how many items

date = np.unique(f_csv['date'])

store_nbr = np.unique(f_csv['store_nbr'])

item_nbr = np.unique(f_csv['item_nbr'])

unit_sales = np.unique(f_csv['unit_sales'])

print("%d days, %d stores, %d items" % (len(date),len(store_nbr),len(item_nbr)))
#Nan to -1

item_id_dict = {}

for item in item_nbr:

    item_id = np.where(f_csv['item_nbr']==item)

    item_id_dict[str(item)] = item_id

tmp_onpromotion = np.zeros(len(f_csv['onpromotion']))

tmp_onpromotion[np.where(f_csv['onpromotion'] != f_csv['onpromotion'])] = -1

tmp_onpromotion[np.where(f_csv['onpromotion'] is True)] = 1

tmp_onpromotion[np.where(f_csv['onpromotion'] is False)] = 0

print("before trans:", f_csv['onpromotion'])

print("after trans:",tmp_onpromotion)