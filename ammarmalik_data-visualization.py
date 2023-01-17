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
%matplotlib inline
NA2018 = pd.read_csv("../input/National Assembly Constituencies 2018.csv", encoding = "ISO-8859-1")
Prov = NA2018[['PROVINCE']]
Punjab = Prov.loc[Prov['PROVINCE']=='Punjab'].shape[0]
Sindh = Prov.loc[Prov['PROVINCE']=='Sindh'].shape[0]
KP = Prov.loc[Prov['PROVINCE']=='Khyber Pakhtunkhwa'].shape[0]
Baloch = Prov.loc[Prov['PROVINCE']=='Balochistan'].shape[0]
Seat_Array = pd.DataFrame([Punjab, Sindh, KP, Baloch], index=['Punjab', 'Sindh', 'Balochistan', 'Khyber Pakhtunkhwa'], columns = ['NA Seat Distribution (%)'])
# make the plot
Seat_Array.plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%.2f')
Population = [112019014, 47866051, 12344408 , 35525047 ]

Pol_Array = pd.DataFrame(Population, index=['Punjab', 'Sindh', 'Balochistan', 'Khyber Pakhtunkhwa'], columns = ['Population Distribution (%)'])
Pol_Array.plot(kind='pie', subplots=True, figsize=(8, 8), autopct='%.2f')
