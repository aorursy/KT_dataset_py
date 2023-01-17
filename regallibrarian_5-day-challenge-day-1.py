#importing libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read in CSV file 

df = pd.read_csv("../input/winemag-data-130k-v2.csv")
#Look at top 5 data entries

df.head(5)
#Summarize data

df.describe()
#check for null values and incomplete data

df.isnull().values.any(), df.shape