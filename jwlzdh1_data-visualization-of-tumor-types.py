import pandas

import numpy as np

import matplotlib.pyplot as plt

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

plt.style.use('ggplot')



%matplotlib inline

%matplotlib inline

BC = pandas.read_csv('../input/data.csv')

BC.head(20)

#PairPlot



seaborn.pairplot(BC.iloc[:,1:11], hue="diagnosis",diag_kind="kde",diag_kws=dict(shade=True))
# Tumor Feature Measurements vs Diagnosis 



# This converts the string characters for diagnosis and converts them to binomial factors. This is needed for a layered histogram to compare values of Malignant and Benign tumors.





BC['diagnosis'] = BC['diagnosis'].map({'M':1,'B':0})







var_mean=list(BC.columns[2:12])

# split dataframe into two based on diagnosis

M=BC[BC['diagnosis'] ==1]

B=BC[BC['diagnosis'] ==0]



plt.rcParams.update({'font.size': 8})

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

axes = axes.ravel()



for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(BC[var_mean[idx]]) - min(BC[var_mean[idx]]))/50

    ax.hist([M[var_mean[idx]],B[var_mean[idx]]], bins=np.arange(min(BC[var_mean[idx]]), max(BC[var_mean[idx]]) + binwidth, binwidth) , 

            alpha=0.5,stacked=True, normed = True, label=['Malignant','Benign'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(var_mean[idx])



    plt.tight_layout()

plt.show()
#Heatmap



import seaborn as sns

corr = BC.iloc[:,1:12].corr()

heat1=sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.show()