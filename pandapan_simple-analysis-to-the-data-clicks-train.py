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
clicked = pd.read_csv("../input/clicks_train.csv")

print(clicked.head())

print("clicks_train Dimension:",clicked.shape)
clicked_1 = clicked[["ad_id","clicked"]].groupby("ad_id").sum()

clicked_1 = clicked_1.sort("clicked",ascending=False)

clicked_1.head()
clicked_2 = pd.DataFrame(clicked[["ad_id"]].groupby("ad_id").size().rename('counts'))

clicked_2 = clicked_2.sort("counts",ascending=False)

clicked_2.head()
click_ana = pd.concat([clicked_1,clicked_2],axis=1)

print(click_ana.shape)

print(click_ana.head())
del clicked_1, clicked_2
click_ana["rate"] = click_ana["clicked"]/click_ana["counts"]
click_ana.head()
click_ana.describe()
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

%matplotlib inline

x = list(click_ana["counts"])

y = list(click_ana["clicked"])

MX = max(x)

MY = max(y)

plt.figure(figsize=(12, 12))

plt.scatter(x,y,color="blue")

plt.xlabel("# of counts")

plt.ylabel("# of cliced")

plt.xlim(0,MX+2)

plt.ylim(0,MY+2)

plt.show()