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
#import some necessary librairies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
train.describe()
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
nulls = train.apply(lambda x: x.isnull().sum())

print(nulls.sum())
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
train.drop(["Soil_Type7","Soil_Type15","Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"], axis = 1, inplace = True)

test.drop(["Soil_Type7","Soil_Type15","Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"], axis = 1, inplace = True)



#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
top_corr_cols = corrmat[abs(corrmat.Cover_Type) >= .1].Cover_Type.sort_values(ascending=False).keys()

print(top_corr_cols)
train.drop(["Wilderness_Area4"], axis = 1, inplace = True)

test.drop(["Wilderness_Area4"], axis = 1, inplace = True)



#correlation matrix

plt.figure()

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
train.drop(["Wilderness_Area2"], axis = 1, inplace = True)

test.drop(["Wilderness_Area2"], axis = 1, inplace = True)



#correlation matrix

plt.figure()

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
y_train = train.Cover_Type.reset_index(drop=True, inplace=False)

x_train = train.drop("Cover_Type", axis = 1, inplace = False)
def model_cross_val(model, param_grid, x_train, y_train):

    from sklearn.model_selection import GridSearchCV

    k_fold = 5

    search = GridSearchCV(model, param_grid=param_grid, n_jobs = 4, cv=k_fold, iid=False)

    search.fit(x_train, y_train)

    

    model = search.best_estimator_

    scoring = 'accuracy'

    from sklearn.model_selection import cross_val_score

    score = cross_val_score(model, x_train, y_train, cv=k_fold, n_jobs= 4, scoring=scoring)

    return round(np.mean(score)*100, 2), model, search.best_params_
rf_param_grid = {"n_estimators": [700, 1000],

                  "max_depth": [30,35,40]}



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators =1000, max_depth = 35)



rf_score, rf_model, rf_best_params = model_cross_val(rf_model, rf_param_grid, x_train, y_train)

print('rf score ', rf_score)

print('rf best params ', rf_best_params)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators =700, max_depth = 40)

rf_model.fit(x_train, y_train)
pred = rf_model.predict(test)
submission = pd.DataFrame({

        "Id": test_ID,

        "Cover_Type": pred

    })



submission.to_csv('submission.csv', index=False)