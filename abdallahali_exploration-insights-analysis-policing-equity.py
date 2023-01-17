# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
path = path = "../input"

filesDict = []
extList = []
for root,d_names,f_names in os.walk(path):
    for f in f_names:
        filesDict.append(os.path.join(root, f))

print("We have = %s" %len(filesDict) ,"files")
print(filesDict[:5])

# extratct all the files extensions
extList = [ ext.split('.')[-1] for ext in filesDict]
print(extList)
# filter the extensions list to remove the _DS_Store
extList = [ ext.split('.')[-1] for ext in filesDict if len(ext.split('.')[-1]) == 3]
print(extList)

# filter the extensions list to get the unique extensions
extType = list(set(extList))
extType
censusData = [csv for csv in filesDict if csv.split('.')[-1] == 'csv']
censusDataANN = [ann for ann in censusData if 'with_ann' in ann]

for x in censusDataANN:
    if 'ACS_education-attainment-over-25' in x:
        y = pd.read_csv(x)
        print(y.shape)
    if 'ACS_race-sex-age' in x:
        y = pd.read_csv(x)
        print(y.shape)
censusDataANN[:5]
# print(censusData[18])
# data = pd.read_csv(censusData[18])
# data.shape

