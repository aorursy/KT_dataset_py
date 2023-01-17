# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



                            #     USING ONE RANDOM FOREST CLASSIFIER



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

import seaborn as sns

from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import math as masampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 



Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv") 

Trainset1 = Trainset1.dropna()



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



dummies1 = pd.get_dummies(Trainset1.Month)

merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = dummies3.drop([1], axis = 'columns')

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

dummies4 = dummies4.drop([9], axis = 'columns')

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

#dummies5 = dummies5.drop([9], axis = 'columns')

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



dummies6 = pd.get_dummies(merged5.Weekend)

merged6 = pd.concat([merged5,dummies6],axis = 'columns')



y = merged6['Revenue']



Trainset = merged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')

Trainset = pd.concat([Trainset,y],axis = 'columns')

Trainset = Trainset.dropna()



y = Trainset['Revenue']



features = Trainset.columns[:55]



xdummies1 = pd.get_dummies(xset.Month)

xmerged1 = pd.concat([xset,xdummies1],axis = 'columns')



xdummies2 = pd.get_dummies(xmerged1.VisitorType)

xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')



xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)

xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')



xdummies4 = pd.get_dummies(xmerged3.Browser)

xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')



xdummies5 = pd.get_dummies(xmerged4.Province)

xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')



xdummies6 = pd.get_dummies(xmerged5.Weekend)

xmerged6 = pd.concat([xmerged5,xdummies6],axis = 'columns')



xset = xmerged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')

print(xset)



id_ = xset['ID']

xset = xset.drop(['ID'], axis = 'columns')



Test = xset



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 1)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 1)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)



# clf = RandomForestClassifier(n_jobs = 2, random_state = 0)



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(Trainset[features],y)

# clf.fit(Trainset[features],y)



preds = rf_random.predict(Test[features])



print("Random forest = ", preds)

YArray = id_.as_matrix(columns=None)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

print(df)

df.to_csv("submission_random.csv", index=False)



# corr = Trainset.corr()



# plt.figure(figsize=(40,40)) 

# # plot the heatmap

# sns.heatmap(corr, 

#         xticklabels=corr.columns,

#         yticklabels=corr.columns, vmin = -1, vmax =1, center = 0, cmap = sns.diverging_palette(0,220,n=200),square = True)