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
# load csv

shop = pd.read_csv("../input/shopee-competition-page/Dummy data.csv")



#rename column

new_names =  {'number': 'new_number'}

shop.rename(columns=new_names, inplace=True)



#add 1 to column

shop['new_number'] += 1

shop



#output to csv

output = pd.DataFrame({'id': shop.id, 'new_number': shop['new_number']})

# save output to csv

output.to_csv('paddlepop_submission.csv', index=False) 