# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the dataset in df variable .
df = pd.read_csv('../input/procurement-notices.csv')
#quick view of missing data and columns datatypes
df.info()
# changing the spaces in the columns' names into underscores as recommended and done by 'Racheal Tatman' 
cols=[]
for i in df.columns:
    cols.append(i.replace ( ' ' , '_'))
    
cols    
    
# assigning the modfied columns to the dataframe columns as values
df.columns = cols
df.columns
# parse the Publication Date   and  Deadline Date columns into datetime datatype.
df.Publication_Date = pd.to_datetime(df.Publication_Date)
df.Deadline_Date=pd.to_datetime(df.Deadline_Date)
%matplotlib inline
import matplotlib.pyplot as plt

#dropping duplicates 
df.drop_duplicates(inplace=True)
df.duplicated().any()
most =df.Country_Name.value_counts().nlargest ( 10)
#most 10 countries/areas that have been included in funded contracts
most
