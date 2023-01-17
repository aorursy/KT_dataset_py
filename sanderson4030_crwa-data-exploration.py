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
Actual_Result_Units = pd.read_csv('/kaggle/input/crwa-samples/CRWA/CRWA/Actual_Result_Units.csv')

Reporting_Result_Units = pd.read_csv('/kaggle/input/crwa-samples/CRWA/CRWA/Reporting_Result_Units.csv')

Results = pd.read_csv('/kaggle/input/crwa-samples/CRWA/CRWA/Results.csv', encoding='latin-1')
# results = Results.QAQC_Status.unique()

Results.groupby('QAQC_Status').count()

prelim = Results[Results['QAQC_Status'] == 'Preliminary']

prelim.sort_values(by=['Date_Collected'],ascending=False)

prelim['Date_Collected'] =  pd.to_datetime(prelim['Date_Collected'])
Results.sort_values(by=['Date_Collected'],ascending=False)

Results['Date_Collected'] =  pd.to_datetime(Results['Date_Collected'])
Results.sort_values(by=['Date_Collected'],ascending=False)
Collection_Method = pd.read_csv('/kaggle/input/crwa-samples/CRWA/CRWA/Collection_Method.csv')

Collection_Method
Waterbody_Name = pd.read_csv('/kaggle/input/crwa-samples/CRWA/CRWA/Waterbody_Name.csv')

Waterbody_Name