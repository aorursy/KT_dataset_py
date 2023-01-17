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
df = pd.read_table('../input/Video_Games_Sales_as_at_22_Dec_2016.csv',sep=',',header=0,index_col=None)

sub = df.dropna()

#sub



sub = sub[ sub['Critic_Count'] > 10 ]

sub = sub[ sub['User_Count'] > 10 ]

sub = sub.drop( ['Name','Global_Sales','Critic_Count','User_Count','Publisher','Developer'],axis=1 )



y_ = sub['NA_Sales']

sub = sub.drop(['NA_Sales'],axis=1)



sub_cat = sub[['Platform','Genre','Rating']]

sub_cont = sub.drop(['Platform','Genre','Rating'],axis=1)



dummies = pd.get_dummies(sub_cat)



final = pd.concat([sub_cont,dummies],axis=1)

final.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import r2_score



#train/test split

# X_train, X_test, y_train, y_test = train_test_split(

#     final, y_, test_size=0.1, random_state=42)



# cv = RepeatedKFold(n_splits=10, n_repeats=10)



# #Ridge hyper

# alpha = np.logspace(-100, 100, 21,base=10)

# param_grid = dict(alpha=alpha)



# grid = GridSearchCV(

#     Ridge(random_state=15,max_iter=5000),

#     param_grid=param_grid, cv=cv, scoring='r2',n_jobs=-1, verbose=1)

lr = LinearRegression()

model = lr.fit(X_train, y_train)
preds = model.predict(X_test)

r2_score(preds,y_test)

model.coef_
for i in zip(model.coef_,final.columns):

    print(i)