# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

import pandas as pd

import seaborn as sns

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectKBest

from scipy.stats import shapiro

from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.sandbox.stats.multicomp import TukeyHSDResults

from statsmodels.graphics.factorplots import interaction_plot

from pandas.plotting import scatter_matrix
iris=load_iris()
iris.target
dataframe_iris=pd.DataFrame(iris.data,columns=['sepalLength','sepalWidth','petalLength','petalWidth'])
dataframe_iris.shape
dataframe_iris1=pd.DataFrame(iris.target,columns=['target'])
dataframe_iris1.shape
scatter_matrix(dataframe_iris[['sepalLength', 'sepalWidth', 'petalLength','petalWidth']],figsize=(15,10))  

plt.show()
ID=[]

for i in range(0,150):

    ID.append(i)
dataframe=pd.DataFrame(ID,columns=['ID'])
dataframe_iris_new=pd.concat([dataframe_iris,dataframe_iris1,dataframe],axis=1)
dataframe_iris_new.columns
fig = interaction_plot(dataframe_iris_new.sepalWidth,dataframe_iris_new.target,

                       dataframe_iris_new.ID,colors=['red','blue','green'], ms=12)
dataframe_iris_new.info()
dataframe_iris_new.describe()
##############################################
##############################################
print(dataframe_iris_new['sepalWidth'].groupby(dataframe_iris_new['target']).mean())
dataframe_iris_new.mean()
##############################################
stats.shapiro(dataframe_iris_new['sepalWidth'][dataframe_iris_new['target']])
##############################################
p_value=stats.levene(dataframe_iris_new['sepalWidth'],dataframe_iris_new['target'])
p_value
##############################################
##############################################
F_value,P_value=stats.f_oneway(dataframe_iris_new['sepalWidth'],dataframe_iris_new['target'])
print("F_value=",F_value,",","P_value=",P_value)
if F_value>1.0:

    print("******SAMPLES HAVE DIFFERENT MEAN******")

else:

    print("******SAMPLES HAVE EQUAL MEAN******")
if P_value<0.05:

    print("******REJECT NULL HYPOTHESIS******")

else:

    print("******ACCEPT NULL HYPOTHESIS******")
##############################################
tukey = pairwise_tukeyhsd(endog=dataframe_iris_new['sepalWidth'], groups=dataframe_iris_new['target'], alpha=0.05)

print(tukey)
##############################################