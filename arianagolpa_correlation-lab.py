# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

top50 = pd.read_csv("../input/imdb-top-50-movies/IMDB Top 50.csv", encoding = "latin-1")
top50.head()
top50[["Rank", "Runtime", "Rating", "Gross_Earning_in_Mil"]].corr()
from scipy import stats

pearson_coef, p_value = stats.pearsonr(top50["Rank"], top50["Runtime"])



print(pearson_coef)



print(p_value)
import seaborn as sns

import matplotlib.pyplot as plt



sns.regplot(x=top50["Rank"], y=top50["Runtime"], marker= "P")

#sns.set_palette("Reds", 8, .75)

plt.show()
