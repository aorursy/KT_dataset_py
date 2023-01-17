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
cereal = pd.read_csv("../input/cereal.csv")
pd.DataFrame.describe(cereal)
cereal[:]
import matplotlib as mp
mp.pyplot.hist(cereal["sugars"])
mp.pyplot.title("grams of sugar per serve")

mp.pyplot.hist(cereal["sugars"])
from scipy.stats import ttest_ind
hot_sugar = cereal.loc[cereal['type'] == 'H']["sugars"]

cold_sugar = cereal.loc[cereal['type'] == 'C']["sugars"]

ttest_ind(hot_sugar, cold_sugar, equal_var = False)
mp.pyplot.title("Hot Cereal - grams of sugar per serve")

mp.pyplot.hist(hot_sugar)
mp.pyplot.title("Cold Cereal - grams of sugar per serve")

mp.pyplot.hist(cold_sugar)