# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Vis Library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



airplane_df = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")



# Change Column headings to uppercase

airplane_df.columns = airplane_df.columns.str.upper()

airplane_df.head(100)



#airplane_df.describe()

x = airplane_df["FATALITIES"]

x = x[x.notnull()]

sns.distplot(x[x <100], kde=False).set_title("Fatality Distribution")
