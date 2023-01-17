# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

from dateutil.parser import parse





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path = "../input/Parking_Violations_Issued_-_Fiscal_Year_2016.csv"

parking = pd.read_csv(path)
parking.columns
#filter data to just 2016

parking_f = parking[parking['Issue Date'].str.contains('2016')]
#Still a lot of rows 

len(parking_f)
columns = ['Summons Number','Issue Date','Registration State', 'Violation Code']

groups = ['Issue Date','Registration State', 'Violation Code']

groups = ['Issue Date','Registration State']



parking_g = parking_f[columns].groupby(groups)

type(parking_g['Summons Number'])
summary1 = parking_g['Summons Number'].nunique()
len(summary1)
summary1.head(100).to_csv('parking_output.csv')
parking_f = parking[parking['year'] == '2016']

type(parking_f)
parking_g = parking.groupby('Issue Date').count()

parking_g