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
#
# Hi I am Practising the Data Science Concepts, below Program is for performing Data Correlation
# and to Identify the Combitations of Minimum Correlating and Maximum Correlating Variables
#
import numpy as np
import pandas as pd
data=np.random.randint(0,100,(17,6))
data[:,0]=np.random.randint(18,50,17)
data[:,1]=np.random.randint(300000,1500000,17)
data[:,2]=np.random.randint(0,15,17)
data[:,3]=np.random.randint(0,10,17)
data[:,4]=np.random.randint(0,5,17)
data[:,5]=np.random.randint(100000,500000,17)
data
data_corrcoef=np.corrcoef(data.T)
data_corrcoef
pd_data_corrcoef=pd.DataFrame(data_corrcoef)
pd_data_corrcoef
data_corrcoef.min()
data_corrcoef.argmin()
data_corrcoef.shape[1]
# Identifying the Variables having min. correlation
MinCorr_var_1=data_corrcoef.argmin()//data_corrcoef.shape[1]
MinCorr_var_2=data_corrcoef.argmin()%data_corrcoef.shape[1]
MinCorr_var_1,MinCorr_var_2
np.eye(data_corrcoef.shape[1])
max_data_corrcoef=data_corrcoef-np.eye(data_corrcoef.shape[1])
pd_max=pd.DataFrame(max_data_corrcoef)
pd_max
max_data_corrcoef.max()
max_data_corrcoef.argmax()
# Identifying the Variables with Maximum Correlation
maxCorr_var_1=max_data_corrcoef.argmax()//max_data_corrcoef.shape[1]
maxCorr_var_2=max_data_corrcoef.argmax()%max_data_corrcoef.shape[1]
maxCorr_var_1,maxCorr_var_2