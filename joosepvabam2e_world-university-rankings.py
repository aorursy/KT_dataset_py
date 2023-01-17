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

import numpy as np
import pandas as pd
%matplotlib inline
pd.set_option("display.max_rows", 20)
df = pd.read_csv("../input/cwurData.csv")
df
df[df["country"] == "Estonia"]
df.groupby("country")["quality_of_education"].mean()
df2= pd.DataFrame(df.groupby("country")["quality_of_education"].mean())
df2.sort_values("quality_of_education", ascending = False)
df3 = df[df["year"] == 2015]
df3.country.value_counts()
df.patents.plot.hist(bins=20, grid=False, rwidth=0.80);
df.plot.scatter("publications", "score", alpha=0.3);