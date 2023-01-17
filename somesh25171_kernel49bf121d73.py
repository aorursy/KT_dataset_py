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
#loading the dataset and printing data of first 5 columns

df =  pd.read_csv("/kaggle/input/coursera-course-data/coursera-course-data.csv")

df.head(5)
#getting information about the dataset.Type of features availble in the dataset

df.info()
#checking for any null values

df.isnull().sum()
#dropping the unnamed column and printing data of first 5 columns

df = df.drop(columns = ['Unnamed: 0'])

df.head(5)
#finding the universities offered different courses from 'Name' column and printing data of first 5 columns

c=[]

for i in df.Name.values:

    c.append((i.rsplit('(')[-1]).replace(')',''))

    d = pd.DataFrame(c,columns = ['Offered_by'])

d.head(5)
#finding the topic of the courses from the 'Link' column and printing data of last 5 columns

a = []

for i in df.Link.values:

    p = (i.lstrip('https://coursera.org/learn/')).replace('-',' ')

    a.append(p)

    b = pd.DataFrame(a,columns = ['Topic'])

    

b.tail(5)
#concatenate the 'Topic' and 'Link' in the main dataframe and printing the data of last 5 columns

df = pd.concat([df,b,d],axis =1)

df.tail(5)