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
# Import and subset dataset into hot/cold cereals 

cereals = pd.read_csv("../input/cereal.csv")

cereals.describe()



hot = cereals.loc[cereals['type'] == "H"] # That took quite a bit of googling

cold = cereals.loc[cereals['type'] == "C"] # Not a straightforward command!
# Histogram of hot cereals' sugar content

import matplotlib.pyplot as plt

plt.hist(hot["sugars"])

plt.title("Histogram of Sugars in Hot Cereals")



# Let's get the text output. Turns out Quaker has a -1 for some reason. 

hot["sugars"].describe()
# Histogram of cold cereals' sugar content

plt.hist(cold["sugars"])

plt.title("Histogram of Sugars in Cold Cereals")



# Let's get the text output. Turns out Quaker has a -1 for some reason. 

cold["sugars"].describe()
# We can't exactly do a t-test with one group with 3 cereals and one group with 74 (totes unfair)

# So let's try a new subset - Kellogg vs. General Mills cold cereals!



kellogg = cold.loc[cold['mfr'] == "K"]

generalmills = cold.loc[cold['mfr'] == "G"]



# Kellogg has 24 cold cereals

plt.hist(kellogg["sugars"])

plt.title("Histogram of Sugars in Kellogg's Cold Cereals")



#kellogg["sugars"].describe()

# General Mills has 22 cold cereals

plt.hist(generalmills["sugars"])

plt.title("Histogram of Sugars in GM's Cold Cereals")



#generalmills["sugars"].describe()

# Now we can do the t-test. I'm going to predict no differences, becasue the

# histograms look similar in both groups.



#scipy.stats import ttest_ind didn't work. But this did...probably needed the whole package



from scipy import stats

stats.ttest_ind(kellogg["sugars"],generalmills["sugars"])



# p = 0.75! No difference.