# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pt

from scipy.stats import ttest_ind



csv = pd.read_csv("../input/cereal.csv")

#print(csv)

#csv.describe()

#pt.hist(csv["sugars"])

#pt.title("Sugar in cereals")

#scipy.stats("")

hot = np.array(csv[(csv.type=="H")].sugars)

cold = np.array(csv[csv.type=="C"].sugars)

ttest_ind(hot, cold, equal_var=False)



#pt.hist(hot)

#pt.hist(cold)



# Any results you write to the current directory are saved as output.