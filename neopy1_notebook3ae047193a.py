# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
arrests = pd.read_csv('../input/BPD_Arrests.csv', dtype= {'Arrest':float, 'Age':float, 'Sex':str,'Race':str,'ArrestDate':dt.datetime,'ArrestTime':dt.datetime,'ArrestLocation':str,'IncidentOffense':str,'IncidentLocation':str,'Charge':object,'ChargeDescription':str,'District':str,'Post':float,'Neighborhood':str,'Location 1':object})

arrests.head(10)
arrests.dtypes
arrests.ArrestDate = pd.to_datetime(arrests['ArrestDate'], format= '%m/%d/%Y' )

arrests['ArrestTime'] = arrests['ArrestTime'].str.replace(r'.',':')
arrests.ArrestTime = pd.to_datetime(arrests['ArrestTime'], format= '%H:%M' )
arrests.ArrestTime
arrests.Race.value_counts()
type(arrests.ChargeDescription.value_counts())
arrests['Hour'] = arrests['ArrestTime'].apply(lambda x: x.hour)
arrests.Hour
arrests.Hour.hist()
arrests.Race.hist()
#arrests.Age.hist(figsize=(8,6)) 

arrests.Age.plot()
import matplotlib.pyplot as plt

fig = plt.figure()