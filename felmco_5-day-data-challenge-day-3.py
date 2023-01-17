# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cereal.csv")
df.head()
df[["sodium","sugars"]].head()
ttest_ind(df["sodium"],df["sugars"],equal_var=False)
df["sodium"].plot(kind='hist',bins=60).set_title('Histogram Sodium')
df["sugars"].plot(kind='hist',bins=60).set_title('Histogram Sugars')