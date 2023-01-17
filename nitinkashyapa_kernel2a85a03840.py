import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot

sns.set(style="darkgrid")
dataset=pd.read_csv("../input/beneficiary-schemes-for-indian-women/WomenBeneficiary.csv")

dataset.hist()

pyplot.show()
sns.lmplot(x = "State/UT", y="Number of Beneficiaries 2015-16",data=dataset) #showing some unknown errors but getting graph at the end so check it out.