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
f = open("/kaggle/input/hashcode-drone-delivery/busy_day.in", "r")

parameters = list(map(int,f.readline().split()))
rows = parameters[0]
cols = parameters[1]
no_drones = parameters[2]
turns = parameters[3]
max_weight = parameters[4]
no_products = int(f.readline())
products = list(map(int,f.readline().split()))
no_warehouses = int(f.readline())
ware_location = np.zeros(shape=(no_warehouses,2))
ware_products = np.zeros(shape=(no_warehouses,no_products))
for warehouse_index in range(no_warehouses):
    
    ware_location[warehouse_index] = list(map(int,f.readline().split()))
    ware_products[warehouse_index] = list(map(int,f.readline().split()))
    
no_orders = int(f.readline())
order_location = np.zeros(shape=(no_orders,2))
order_items = np.zeros(shape=(no_orders,1))
order_type = np.zeros(shape=(no_orders,no_products))
for order_index in range (no_orders):
    order_location[order_index] = list(map(int,f.readline().split()))
    order_items[order_index] = int(f.readline())
    #TODO - load the order type
    odr_type = f.readline()
    #order_type[order_index] = list(map(int,f.readline().split()))
