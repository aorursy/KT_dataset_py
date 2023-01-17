%matplotlib inline



import sklearn



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt



import csv

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, make_scorer, mean_squared_error



Xmodelmake = pd.read_csv("../input/data.csv")

print(Xmodelmake.describe())



hp=[]



Years=Xmodelmake.Year.unique()

#Years=list.sort(Years.tolist())

#Years=pd.DataFrame(Years)

for y in Years:

    #print(y,Xmodelmake[Xmodelmake['Year']==y].mean())

    hp.extend(Xmodelmake[Xmodelmake['Year']==y].mean())

    

    

#print(hp)  

hp=np.reshape(hp, (28,-1))      

hp = pd.DataFrame(hp)

hp.columns=(['year', 'enginehp','cylind','doors','mpg','city','popular','MSRP'])    

hp=hp.sort(['year'])

hp['after']=(hp['year']>2007)*1

#print(hp)
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns





# Compute the correlation matrix





g = sns.pairplot(hp[['year', 'enginehp','after']], hue='after',size=3)

g.set(xticklabels=[])





g = sns.pairplot(hp[['year','mpg','city','after']], hue='after',size=3)

g.set(xticklabels=[])



g = sns.pairplot(hp[['year','mpg','popular','after']], hue='after',size=3)

g.set(xticklabels=[])



corr = hp.corr()

ax = sns.heatmap(corr)