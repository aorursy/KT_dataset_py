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
# Read in 2017 Data

data_2017 = pd.read_csv("../input/20170308hundehalter.csv")

# Read in 2016 Data

data_2016 = pd.read_csv("../input/20160307hundehalter.csv")

# Read in 2015 Data

data_2015 = pd.read_csv("../input/20151001hundehalter.csv")
# Summarize 2017 Data

data_2017.describe()
# Summarize 2016 Data

data_2016.describe()
# Summarize 2015 Data

data_2015.describe()
# Read in the last data set

data = pd.read_csv('../input/zuordnungstabellehunderassehundetyp.csv')

# Summarize it

data.describe()