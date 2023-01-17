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

import seaborn as sns

import scipy as sp

import numpy as np
df = pd.read_csv("../input/scrubbed.csv")
df.head()
# Set context to `"paper"`

sns.set_context("paper", font_scale=2, rc={"font.size":8,"axes.labelsize":2})



sns.countplot(df["state"])
df.groupby('state').count().plot(kind='bar')