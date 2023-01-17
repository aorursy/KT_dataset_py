# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
raw_df = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")
## raw_df.head()
## raw_df.describe()

# Query by Country = Spain
spa_df = raw_df[['dt','AverageTemperature','AverageTemperatureUncertainty']][raw_df['Country'] == "Spain"]
## spa_df.head()
## spa_df.describe()
print ("Raw Spanish Elements {}".format(len(spa_df)))

# Selecting Finite Values
spa_df = spa_df[(np.isfinite(spa_df['AverageTemperature'])) & (np.isfinite(spa_df['AverageTemperatureUncertainty']))]
print ("Selected Spanish Elements {}".format(len(spa_df)))

dates = spa_df['dt']
values = spa_df['AverageTemperature']


