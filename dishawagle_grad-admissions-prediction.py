# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

datav2 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

test = datav2[-100:]

val = datav2[:-100]

data = data.append(val)
data.head()
data.describe()
import matplotlib.pyplot as plt

import seaborn as sns

corr = data.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f,ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, cmap=cmap, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink":.5})
list(data)
ax1 = sns.distplot(data['Chance of Admit '])
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV
def preprocess(data):

    data.drop('Serial No.',axis=1)

    return data
data = preprocess(data)
X, y = data.drop('Chance of Admit ',axis=1),data['Chance of Admit ']

model = RandomForestRegressor(n_estimators=10)

param_grid = {'model__n_estimators':[5,10,50, 100,150], 'model__random_state':[0,0.2,0.4], 'model__max_depth':[2,4,6,8,10]}

pipe = Pipeline(steps=[('model',model)])

search = GridSearchCV(pipe, param_grid, n_jobs=-1)

search.fit(X,y)
search.best_params_
mymodel = RandomForestRegressor(max_depth=6,n_estimators=50,random_state=0)

my_pipeline = Pipeline(steps=[('model',mymodel)])

scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5, scoring = 'neg_root_mean_squared_error')
scores.mean()
my_pipeline.fit(X,y)
val_predictions = my_pipeline.predict(X)
from sklearn.metrics import mean_squared_error 
val_score = mean_squared_error(y, val_predictions, squared=False)
val_score
test = preprocess(test)

test_X , test_y = test.drop('Chance of Admit ',axis=1),test['Chance of Admit ']

test_predictions = my_pipeline.predict(test_X)

test_score = mean_squared_error(test_y, test_predictions, squared=False)

test_score