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
pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv')
data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding='latin-1')
data.head()
data.describe()
data["INCIDENT_NUMBER"]
data.loc[:,"INCIDENT_NUMBER"]
data.iloc[:,1]
data["INCIDENT_NUMBER"][0]
data.loc[0,"INCIDENT_NUMBER"]
data.iloc[0,0]
data.loc[0,:]
data.iloc[0,:]
data.loc[:10,"OFFENSE_CODE_GROUP"]
data.iloc[:10,2]
data.iloc[:,[2, 4, 6, 8]]
data.loc[[2, 4, 6, 8], ["OFFENSE_CODE_GROUP", "DISTRICT", "SHOOTING", "YEAR"]]
data.loc[:9, ["OFFENSE_CODE_GROUP", "DISTRICT", "SHOOTING", "YEAR"]]
data.iloc[:10, [2, 4, 6, 8]]
data.loc[data.OFFENSE_CODE_GROUP == "Vandalism"]
data.loc[data["OFFENSE_CODE_GROUP"] == "Vandalism"]
data.loc[(data["OFFENSE_CODE_GROUP"] == "Vandalism") & (data["DAY_OF_WEEK"] == "Monday")]
data.loc[data["OFFENSE_CODE_GROUP"] == "Auto Theft"].loc[:,"DAY_OF_WEEK"].value_counts().plot.bar()