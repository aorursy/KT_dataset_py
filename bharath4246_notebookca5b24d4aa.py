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
import pandas as pd

import numpy as np

import matplotlib as plt

df = pd.DataFrame({"int_col" : [1,4,3,6,0, None], "float_col" : [0.2,0.5,0.9, None, 0.8, 2.2], "str_col" : ["a", "b", "c", "d", "e", "f"]})

print(df)

#df.ix[:, ["int_col", "str_col"]]

df[["int_col", "float_col"]]



df[df["float_col"] > 0.8]



df[df["float_col"] == 0.9]

print(df)

df[(df["float_col"] > 0.4) | (df["int_col"] > 3)]