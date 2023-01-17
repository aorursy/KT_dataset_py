# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")

data.head()
for col_name in data.columns:

    print(col_name)
physicfor1000=data[["Country_Region","Physicians_per_1000_2009-18"]].sort_values(by="Physicians_per_1000_2009-18",ascending=False)

physicfor1000[0:20]
physicfor1000[80:100]
surgeons=data[["Country_Region","Specialist_surgical_per_1000_2008-18"]].sort_values(

    by="Specialist_surgical_per_1000_2008-18",ascending=False)

surgeons[0:20]
surgeons[20:45]
surgeons[45:70]
health_except=data[["Country_Region","Health_exp_public_pct_2016"]].sort_values(

    by="Health_exp_public_pct_2016",ascending=False)



health_except[0:28]


all_money_per_capita=data[["Country_Region","Health_exp_per_capita_USD_2016"]].sort_values(

    by="Health_exp_per_capita_USD_2016",ascending=False)



all_money_per_capita[0:28]
all_money_per_capita[28:50]
data.isnull().sum()
del data["Province_State"]
correlations=data.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlations,annot=True)
correlations=data[["Physicians_per_1000_2009-18",

"Nurse_midwife_per_1000_2009-18",

"Specialist_surgical_per_1000_2008-18",]].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(correlations,annot=True)


correlations=data[["Physicians_per_1000_2009-18",

"Health_exp_pct_GDP_2016",

                   "Health_exp_public_pct_2016"]].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(correlations,annot=True)


correlations=data[["Physicians_per_1000_2009-18",

"Health_exp_out_of_pocket_pct_2016",

"Health_exp_per_capita_USD_2016",

"per_capita_exp_PPP_2016"

                  ]].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(correlations,annot=True)
withoutNA=data.dropna()

x=withoutNA[["Health_exp_pct_GDP_2016",

"Health_exp_public_pct_2016",

"Health_exp_out_of_pocket_pct_2016",

"Health_exp_per_capita_USD_2016",

"per_capita_exp_PPP_2016",

"External_health_exp_pct_2016"

]]



y=withoutNA[["Physicians_per_1000_2009-18"]]



from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=144,max_depth=2)

regressor.fit(x,y)

regressor.score(x,y)
from sklearn import tree

from sklearn.tree import export_graphviz

export_graphviz(regressor,out_file='tree_limited.dot',feature_names=x.columns)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=200

from IPython.display import Image

Image(filename = 'tree_limited.png')
regressor=DecisionTreeRegressor(random_state=144,max_depth=3)

regressor.fit(x,y)

regressor.score(x,y)

export_graphviz(regressor,out_file='tree_limited.dot',feature_names=x.columns)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=200

from IPython.display import Image

Image(filename = 'tree_limited.png')
withoutNAsmaller7=withoutNA[withoutNA["Physicians_per_1000_2009-18"]<=6.5]

x=withoutNAsmaller7[["Health_exp_pct_GDP_2016",

"Health_exp_public_pct_2016",

"Health_exp_out_of_pocket_pct_2016",

"Health_exp_per_capita_USD_2016",

"per_capita_exp_PPP_2016",

"External_health_exp_pct_2016"

]]



y=withoutNAsmaller7[["Physicians_per_1000_2009-18"]]

regressor=DecisionTreeRegressor(random_state=144,max_depth=3)

regressor.fit(x,y)

regressor.score(x,y)

export_graphviz(regressor,out_file='tree_limited.dot',feature_names=x.columns)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=200

from IPython.display import Image

Image(filename = 'tree_limited.png')