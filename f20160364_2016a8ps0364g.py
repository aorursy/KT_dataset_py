# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
train = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
y_train = train['rating'] 

y = y_train
df = pd.concat([train.drop('rating',axis=1),test],axis=0)



numerical_features = [x for x in df.columns if x not in ['type']] 

categorical_features = ['type']
df.describe()
df.fillna(df.mean(),inplace= True)
corr=train.corr()

plt.figure(figsize=(12,9))

mask = np.zeros_like(corr)

cmap=sns.diverging_palette(220,10,as_cmap=True)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

  ax = sns.heatmap(corr,cmap=cmap,mask=mask, vmax=.3, square=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['type']= le.fit_transform(df['type'])
num_features = ['feature5']

x_train = df.drop(['feature5'],axis=1)[0:4547]

x_test = df.drop(['feature5'],axis=1)[4547:]
from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier() 

parameters = {'n_estimators':[500,600,700],'max_depth':[90,100,110,120],'bootstrap':[True,False],'min_samples_split':[2,3,4,5],'min_samples_leaf':[2,3,4,5],'max_features':[None,'sqrt']}  

scorer = make_scorer(mean_squared_error)        



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)        

grid_fit = grid_obj.fit(x_train,y_train)        



best_clf = grid_fit.best_estimator_
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression



# Best model

clf1 = RandomForestRegressor(n_estimators=3000,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',max_depth=15,bootstrap=False).fit(x_train,y_train) 



# Second best model

clf2 = RandomForestRegressor(n_estimators=1400,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',max_depth=15,bootstrap=False).fit(x_train,y_train) 



y_pred_1 = clf1.predict(x_test)

y_pred_2 = clf2.predict(x_test)



y_pred_1 = [int(round(x)) for x in y_pred_1]

y_pred_2 = [int(round(x)) for x in y_pred_2]
answer = pd.DataFrame(data={'id':df[4547:]['id'],'rating':y_pred_1})

answer.to_csv('submission1.csv',index=False)



answer = pd.DataFrame(data={'id':df[4547:]['id'],'rating':y_pred_2})

answer.to_csv('submission2.csv',index=False)