# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

ta = pd.read_csv("../input/Traffic accidents by month of occurrence 2001-2014.csv")

print (ta.head())

print (ta.tail())

print (ta.info())

print (ta.describe())

# Any results you write to the current directory are saved as output.



print("Which state had the maximum accidents?")

print("Which year had the maximum accidents?")

print("Is there a relation between months and addicents?")

print("Which is the safest state?")

print("Which are more in number ? Railway or Raod accidents")
