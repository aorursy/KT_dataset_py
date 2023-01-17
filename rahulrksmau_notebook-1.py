# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/up_res.csv')

population = data['votes'].sum()

#print (data.head())



#Label Encoding of Data



from sklearn import preprocessing

features= ['seat_allotment', 'ac', 'district', 'party']

for feature in features:

    le = preprocessing.LabelEncoder()

    le = le.fit(data[feature])

    data[feature] = le.transform(data[feature])



print (data.head())

print (data.groupby('seat_allotment').size())

print (data.groupby('ac').size())

print (data.groupby('district').size())

print (data.groupby('party').size())