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
df= pd.read_csv("../input/cereal.csv")
df.describe()
x = df['calories']
import seaborn as sns

#sns.set(color_codes = True)

#sns.distplot(x,bins = 20, kde=False, rug = True) # kde is Kernel Density Estimate

#rug is small vertical tick at each observation

# ".set_title('title') will add title to the plot

sns.distplot(x,kde = False).set_title("calories")