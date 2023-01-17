# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

# The path below is based on the output in the previous code cell.

path = "../input/melbourne-housing-snapshot/melb_data.csv"

df = pd.read_csv(path)
print(df.shape)

print(df.size)

print(df.ndim)

print(len(df.columns))
df.columns
df.head()
df.describe()
df.describe(include="all")