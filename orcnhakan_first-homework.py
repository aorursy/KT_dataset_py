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
data = pd.read_csv('../input/heart.csv')



data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row





data3 = data['age'].head()

data4= data['sex'].head()

conc_data_col = pd.concat([data3,data4],axis =1) # axis = 0 : adds dataframes in row

conc_data_col



#concat_data = pd.concat[data1,data2]

data.head(15)
age = data['age']

age1 = data.age>50

print(age1)

data[np.logical_and(data['age']>40, data['trestbps']==100 )]
dictionary = {'UNV':'Kocaeli University','DEPARTMENT' : 'Computer Science'}

dictionary['CİTY']='Kocaeli'



dictionary



for key,value in dictionary.items():

     print(key," : ",value)
data.dtypes
data.info() # sorunlu veri bulunmamaktadır.