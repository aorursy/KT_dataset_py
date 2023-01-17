import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV
inp = pd.read_csv("../input/training/train.csv").dropna()

inp['agent'] = (inp['id']-1)%7

inp.drop(columns=['id','a0','a1','a2','a3','a4','a5','a6'],inplace=True)

display(inp)
data =[]

for i in range(7):

#     display(inp[inp.agent==i])

    data.append(inp[inp.agent==i])

    data[i].drop(columns=['agent'],inplace=True)

display(data)

corr = data[0].corr() 

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(inp.corr(), mask=np.zeros_like(inp.corr(), dtype=np.bool), square=True, ax=ax, annot = False)


%%time

n_estimators = [10,100,5]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [3000,4000]

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

# Random search of parameters

op = data[i].label

inp = data[i].drop(columns=['label'])



rfc_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 5, cv = 3, n_jobs = -1,)



rfc_random.fit(inp, op)









# model.fit(inp, op)



# scores = cross_validate(model, inp, op, cv=3,return_train_score=True,n_jobs=-1

#                         ,scoring='neg_root_mean_squared_error'

#                        )

# print("agent "+str(i)+" : "+str(-scores['test_score'].mean())+" "+str(-scores['train_score'].mean()))

# scores
print(rfc_random.best_params_)
%%time

model = RandomForestRegressor()

for i in range(7):

    op = data[i].label

    inp = data[i].drop(columns=['label'])

    model.fit(inp, op)

    scores = cross_validate(model, inp, op, cv=3,return_train_score=True,n_jobs=-1

                            ,scoring='neg_root_mean_squared_error'

                           )

    print("agent "+str(i)+" : "+str(-scores['test_score'].mean())+" "+str(-scores['train_score'].mean()))