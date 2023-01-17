### 1. Read Data
from IPython.core.display import display, HTML
display(HTML('<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d690937.2663333984!2d-122.36379811452582!3d47.43195624174192!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x54905c8c832d7837%3A0xe280ab6b8b64e03e!2sKing+County%2C+WA%2C+USA!5e0!3m2!1sen!2sin!4v1534150510023" width="1400" height="800" frameborder="0" style="border:0" allowfullscreen></iframe>'))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/kc_house_data.csv")
train.head(10)
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
train.hist(ax = ax)
g = sns.FacetGrid(train, col="yr_built",col_wrap=6)
g.map(sns.distplot, "price")
g = sns.FacetGrid(train, col="yr_renovated",col_wrap=6)
g.map(sns.distplot, "price")
from IPython.display import IFrame
IFrame('https://public.tableau.com/shared/9PN4W4SGK?:display_count=yes?:embed=y&:showVizHome=no', width=1000, height=925)
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/Priceperbedrooms/PriceComparisionwithDifferentParameter?:embed=y&:showVizHome=no', width=1050, height=925)
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/YearBuiltvsRenovatedByPrice/YearBuiltandRenovatedByPrice?:embed=y&:showVizHome=no', width=1000, height=925)
# https://public.tableau.com/views/YearBuiltvsRenovatedByPrice/YearBuiltandRenovatedByPrice?:embed=y&:display_count=yes&publish=yes
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/GradeByPriceMap/GradebyPrices?:embed=y&:showVizHome=no', width=1000, height=925)
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/WaterfrontbyPrice/WaterfrontbyPrice?:embed=y&:showVizHome=no', width=1000, height=925)

# https://public.tableau.com/views/WaterfrontbyPrice/WaterfrontbyPrice?:embed=y&:display_count=yes
