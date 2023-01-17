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
!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip
dataset = pd.read_csv("/kaggle/working/train.csv", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)

features = pd.read_csv("/kaggle/working/features.csv",sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')
dataset.head()
features.head()
stores.head()