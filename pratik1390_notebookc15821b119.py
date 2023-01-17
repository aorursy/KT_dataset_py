# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dftrain = pd.read_csv('../input/train.csv')

dftest = pd.read_csv('../input/test.csv')



# Any results you write to the current directory are saved as output.
dftest.info()
corr=dftrain.corr()

varCorrWithSalePric = corr["SalePrice"]

Threshold = 0.3

highCorrVars = varCorrWithSalePric[(varCorrWithSalePric.values > Threshold) | (varCorrWithSalePric.values < -Threshold) ]
highCorrVars = highCorrVars.index.tolist()

check = np.log1p(dftrain.loc[:,highCorrVars])

check.hist()

#highCorrVars = highCorrVars.index.tolist()

#dftrain["SalePrice"].hist()

plt.figure()

dftrain["GarageYrBlt"].hist()

#plt.hist(np.log(dftrain["YearBuilt"]),color="g", alpha = 0.2)

#plt.hist(np.log(dftrain["GarageYrBlt"]),color="b", alpha = 0.2)
plt.close("all")

#sns.jointplot(x=np.log(dftrain["SalePrice"]), y=dftrain["YearBuilt"],kind="kde",marker="+",color="red")

#sns.regplot(x=dftrain["SalePrice"], y=dftrain["GarageYrBlt"].log(),marker=".",color="green")

sns.regplot(x=np.log(dftrain["SalePrice"]), y=dftrain["GarageYrBlt"],marker=".",color="green")
type(corr[l][a])
for l in corr:

	for a in corr:

		if (corr[l][a]>0.7 and corr[l][a] <1.0) and ((l != "SalePrice") & (a != "SalePrice")):            print(corr[l][a], l, a)
dftrain.info()
import seaborn as sns

sns.heatmap(corr, vmax=.8, square=True)

numeric_feats = dftrain.dtypes[dftrain.dtypes != "object"].index

numeric_feats
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])