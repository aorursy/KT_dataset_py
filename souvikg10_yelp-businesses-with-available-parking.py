# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
yelp = pd.read_csv('../input/yelp_business_attributes.csv')

yelp_location = pd.read_csv('../input/yelp_business.csv')

yelp_business = pd.merge(yelp,yelp_location,on='business_id',how='outer')

parkingwithgarage = yelp_business.loc[yelp_business['BusinessParking_garage']=='True']
parkingwithstreet = yelp_business.loc[yelp_business['BusinessParking_street'] == 'True']
print('Parking with Garage='+ str(parkingwithgarage['BusinessParking_garage'].count()))
print('Parking with Street='+ str(parkingwithstreet['BusinessParking_street'].count()))

print(parkingwithgarage.head())
