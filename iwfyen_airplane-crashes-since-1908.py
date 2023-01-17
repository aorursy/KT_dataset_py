# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read and preprocess ALL the input data.

#However there is no need for preprocessing.

AirCrashPd = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv',sep=',')
# Q1. Yearly how many planes crashed? how many people were on board? how many survived? how many died?

subframe = AirCrashPd[['Date','Aboard','Fatalities']]

subframe['Year'] = subframe['Date'].apply(lambda x: int(str(x)[-4:]))

yeardate = subframe[['Year','Aboard','Fatalities']].groupby('Year').agg(['count','sum'])



fig_yearly,(axy1,axy2)=plt.subplots(2,1,figsize=(15,10))

yeardate['Aboard','sum'].plot(kind='bar',title='Aboard per year',grid=True,ax=axy1,rot=45)

yeardate['Fatalities','sum'].plot(kind='bar',title='Fatalities per year',grid=True,ax=axy2,rot=45)

plt.tight_layout()