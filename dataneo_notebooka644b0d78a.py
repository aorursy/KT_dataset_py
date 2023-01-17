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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



accidents = pd.read_csv("../input/Accidents0515.csv")





casualties = pd.read_csv("../input/Casualties0515.csv",nrows=1)

casualties_cols = list(casualties.columns)

casualties_cols.append('New_Column_2015')

casualties = pd.read_csv("../input/Casualties0515.csv",skiprows=1, header=None, names=casualties_cols)



vehicles = pd.read_csv("../input/Vehicles0515.csv",nrows=1)

vehicles_cols = list(vehicles.columns)

vehicles_cols.append('New_Column_2015')

vehicles = pd.read_csv("../input/Vehicles0515.csv",skiprows=1, header=None, names=vehicles_cols)



vehicles.fillna(0,inplace=True)

casualties.fillna(0,inplace=True)



pd.isnull(vehicles)