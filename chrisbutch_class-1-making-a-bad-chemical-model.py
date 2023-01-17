%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

import pandas as pd

import numpy as np
import os

os.getcwd()
data=pd.read_csv('/kaggle/input/chemical-structure-and-logp/logP_dataset.csv', header=None)
len(data)
data.head(10)
data.columns=['SMILES','LogP']
data.head(10)
import seaborn as sns

ax = sns.violinplot(x=data["LogP"])
set(''.join(data['SMILES']))
import re



regex = re.compile('[^a-zA-Z]')

#First parameter is the replacement, second parameter is your input string

regex.sub('', 'ab3d*E')

#Out: 'abdE'
data['Simple_SMILES']=data['SMILES'].apply(lambda x: regex.sub('',x))
data.head(10)
set(''.join(data['Simple_SMILES']))
elements=[*set(''.join(data['Simple_SMILES']))]
for element in elements:

    data[element]=data['Simple_SMILES'].apply(lambda x:sum(map(lambda y: 1 if element in y else 0, x)))
data.head(10)
import statsmodels.api as sm

X = data[elements]

y = data["LogP"]

model = sm.OLS(y, X).fit()

data["pred_LogP"] = model.predict(X)
data.head(10)
from scipy import stats

def r2(x, y):

    return stats.pearsonr(x, y)[0] ** 2



sns.jointplot(x=data["pred_LogP"],

            y=data["LogP"], 

            kind="reg", 

            stat_func=r2,

            scatter_kws={"s": 1});
allchars=[*set(''.join(data['Simple_SMILES']))]

for char in allchars:

    data[char]=data['Simple_SMILES'].apply(lambda x:sum(map(lambda y: 1 if char in y else 0, x)))
X = data[allchars]

y = data["LogP"]

model = sm.OLS(y, X).fit()

data["pred_LogP2"] = model.predict(X)
sns.jointplot(x=data["pred_LogP2"],

            y=data["LogP"], 

            kind="reg", 

            stat_func=r2,

            scatter_kws={"s": 1});