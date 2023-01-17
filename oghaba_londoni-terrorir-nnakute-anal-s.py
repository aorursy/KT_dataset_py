# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



%matplotlib inline

pd.set_option("display.max_rows", 20)



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')
df
df.info()
df["country_txt"].value_counts()

uk = df[df["country_txt"] == "United Kingdom"]
uk
ukl = uk[uk["city"] == "London"]

ukl
ukl.iyear.plot.hist()
ukl.plot.scatter("imonth", "iday", alpha=0.2)
ukl.groupby(["iyear", "gname"])["nwound", "propvalue"].max()