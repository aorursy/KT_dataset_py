# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotnine
from plotnine import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
plotnine.options.figure_size = (15, 10)
shootings = pd.read_csv("../input/pah_wikp_combo.csv")
# Preprocess
shootings = shootings.drop(columns=["Desc"])
shootings = shootings.dropna(subset=["Wounded"])
shootings["School"] = shootings["School"].fillna("-")
shootings = shootings[shootings["School"] != "-"]
shootings["Date"] = pd.to_datetime(shootings["Date"])
shootings.head()
plot = (ggplot(shootings, aes(x="Date", y="Wounded", color="School", size="Fatalities"))
    + geom_point()
    + ggtitle("School Shootings in the US"))
plot
plot.save("School Shootings in the US.png", width=10, height=5, units="in", dpi=300)
plot = (ggplot(shootings, aes(x="Date", y="Wounded", color="School", size="Fatalities"))
    + geom_point()
    + facet_wrap("School")
    + ggtitle("School Shootings in the US"))
plot
plot.save("School Shootings in the US by School Type.png", width=15, height=10, units="in", dpi=300)