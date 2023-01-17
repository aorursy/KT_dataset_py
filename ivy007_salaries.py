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
import pandas as pd
sal = sal = pd.read_csv('../input/Salaries.csv')
import numpy as np
sal.info()
# Let's try to execute mean() function directly
# It wont be working direct as in the Udemy exercise, becuse the dataset contains many value such as 'Not Provided' ,'0.0', etc, 
#which makes it difficult to apply any arithmetic operations on it.
sal['BasePay'].mean()
sal['BasePay']
# Change value Not Provided to Nan
sal[sal['BasePay'].str.contains('Not Provided' , na =False)] = np.NaN
sal['BasePay'].mean()
sal['BasePay'].type()
#converting the datatype to float \
sal['BasePay'] = sal['BasePay'].apply(lambda x:float(x))
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal['OvertimePay']

sal[~sal['OvertimePay'].str.contains('Not Provided', na = False) ]['OvertimePay'].apply(lambda x : float(x)).max()
sal[sal['EmployeeName'] =='JOSEPH DRISCOLL']['JobTitle']

sal[sal ['EmployeeName'] == 'JOSEPH DRISCOLL'] ['TotalPayBenefits']
sal.iloc[sal['TotalPayBenefits'].idxmax()]
sal.iloc[sal['TotalPayBenefits'].idxmin()]
a = sal.groupby('Year')
a['BasePay'].mean()
sal['JobTitle'].nunique()


