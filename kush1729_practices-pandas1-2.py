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
import numpy as np
# np.linspace(start, stop, num=50, endpoint =True, retstep= False, dtype = None, axis = 0)
sample_50 = np.linspace(1,10)
sample_50[:10]
import pandas as pd
series_1 = pd.Series(sample_50)
print(series_1.head())
# with num =  5
samples = np.linspace(1,10,5)
samples
# Creating another series 
sample_5 = pd.Series(samples)
print(sample_5)
random = np.random.rand(10)
random

ran_ser = pd.Series(random)
print(ran_ser)
dict = {'India': 'New Delhi' , 'Japan': 'Tokoyo' , 'UK': 'London'}
dict_ser = pd.Series(dict)
print(dict_ser)
# Created index and values for above dict series
print(dict_ser.index)
print(dict_ser.values)
# Creates series with array and explicit index
ran = np.arange(10)
ra_ser = pd.Series(ran, index=['A','B','C','D','E','F','G','H','I','J'])
ra_ser
ra_ser.shape
ra_ser.size
# Slicing series from 4 to end and step = 3 
ra_ser[4::3]
#Slicing the series into revrse order
ra_ser[::-1]
#Converting series into back to lists
print(type(ra_ser))
print(type(ra_ser.tolist()))
# Appending the one series to other series
data = ['50','100', '70', '2','5']
da = pd.Series(data)
tmp = ['500','600','700']
da1 = pd.Series(tmp)
new = da.append(da1)
new
#Creating dataframe for performing some basic operations 
exam_data = {'name': ['Arun','Rama','Kanthraj','James','Emily','Michael',
                     'Mathew','Laura','Kevin','Jonas'],
             'score':[12,10,17,np.nan,9,30,15,np.nan,8,19],
            'attempts':[1,3,2,3,2,3,1,1,2,1],
            'qualify': ['yes','no','yes','no','no','yes','yes','no','no','yes']}
labels = ['a','b','c','d','e','f','g','h','i','j']
exam_df = pd.DataFrame(exam_data, index= labels)
exam_df
exam_df.info()
# Students whose score is greater than 12
exam_df[exam_df.score > 12]
# Score with No null value
st_withsc = exam_df[ exam_df.score.notnull()]
st_withsc
# students whose Score with Null 
st_without = exam_df[exam_df['score'].isnull()]
st_without
qual = exam_df[exam_df['qualify'] =='yes']
qual
nqual = exam_df[exam_df['qualify'] != 'yes']

nqual
#Creating New dataframe without score column
nexam = exam_df[['name','attempts','qualify']]
nexam
# find out students who attenmted i time only
att = exam_df['attempts']
att
att.value_counts()
exam_df.loc[exam_df['attempts'] == 1]
# get the max score 
sc = exam_df['score'].max()
sc
