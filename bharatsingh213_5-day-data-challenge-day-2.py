# Import all the Libraries for you applicaiton

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn  as sns # for Drawing chart



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) 
# Load the data to dataframe from CSV file dataset

dt=pd.read_csv("../input/bar_locations.csv");
# Print first five data

print(dt.head());



# Print  data summery for data

print(dt.describe());
# Read a specific column data if data isNumeric

data=dt["Incident Zip"]

print(data)
hist=sns.distplot(data, kde=False);

hist.set_title("Bar Location Info");