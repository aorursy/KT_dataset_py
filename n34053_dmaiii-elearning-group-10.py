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
# Creating dataframe to be able to filter and sort value.



data_a = {"Customer_ID":pd.Series([1,2,3,4,5,6]),"Name":pd.Series(["Huber","Maier","Binder","Unfried","Mustermann","Lehmann"]),"City":pd.Series(["Vienna", "Sydney", "Krems", "Klosterneuburg", "Amsterdam", "Prague"]),"Sum":pd.Series(["5","12","15","8","11","17"]),"Age":pd.Series(["25","18","35","41","52","19"])}

df_a = pd.DataFrame(data_a)
# Show the first 6 rows of the dataframe



df_a.head(6)
# Find out the maximum age.



data_a['Age'].max()
# Print the row with maximum age.



print(df_a[df_a.Age == df_a.Age.max()])
# Print the row with minimum age.



print(df_a[df_a.Age == df_a.Age.min()])