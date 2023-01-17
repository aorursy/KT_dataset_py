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
demo = pd.read_csv("/kaggle/input/yeh-concret-data/Concrete_Data_Yeh.csv")
demo.head()
from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(demo, diag_kind='kde')
sns.heatmap(demo.corr(), annot=True)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
X = demo.drop('csMPa', axis=1)
y = demo['csMPa']
gb.fit(X,y)
gb.feature_importances_
### Iteration 1 - simple linear model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X = demo.drop('csMPa', axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)
lr = LinearRegression()

lr.fit(Xtrain,ytrain)
print("Training R2")
print (lr.score(Xtrain,ytrain))
print("Testing R2")
print (lr.score(Xtest,ytest))
### Iteration 2 - considering only important features
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X = demo[['cement','age','water']]
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)
lr = LinearRegression()

lr.fit(Xtrain,ytrain)
print("Training R2")
print (lr.score(Xtrain,ytrain))
print("Testing R2")
print (lr.score(Xtest,ytest))

### Iteration 3 - Polynomial Features
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

X = demo.drop('csMPa', axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)
poly = PolynomialFeatures(degree=1)

sc = StandardScaler()
scaledXtrain = sc.fit_transform(Xtrain)
scaledXtest = sc.fit_transform(Xtest)

polyXtrain = poly.fit_transform(Xtrain)
polyXtest = poly.transform(Xtest)

lr = LinearRegression()
lr.fit(polyXtrain,ytrain)

lr.fit(Xtrain,ytrain)
print("Training R2")
print (lr.score(Xtrain,ytrain))
print("Testing R2")
print (lr.score(Xtest,ytest))

### Iteration 4 - with Pipeline
from sklearn.pipeline import Pipeline

X = demo.drop('csMPa', axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)

pipe = Pipeline((
    ("sc" , StandardScaler()),
    ("poly",PolynomialFeatures(degree=1)),
    ("lr", LinearRegression())
))

pipe.fit(Xtrain,ytrain)
print("Training R2")
print (pipe.score(Xtrain,ytrain))
print("Testing R2")
print (pipe.score(Xtest,ytest))
### Iteration 5 - transformation Pipeline
from sklearn.pipeline import Pipeline

X = demo.drop('csMPa', axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)

pipe = Pipeline((
    ("sc" , StandardScaler()),
    ("poly",PolynomialFeatures(degree=3))
))

prepTrain = pipe.fit_transform(Xtrain)
prepTtest = pipe.fit_transform(Xtest)

lr = LinearRegression()
lr.fit(prepTrain,ytrain)

print("Training R2")
print (lr.score(prepTrain,ytrain))
print("Testing R2")
print (lr.score(prepTtest,ytest))
#Kfold CV
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X = demo.drop('csMPa',axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)
lr = LinearRegression()

scoresdt = cross_val_score(lr,Xtrain, ytrain, cv=10)
print(scoresdt)
print("Average R2 ",np.mean(scoresdt))
print("SD of accuracy ", np.std(scoresdt))
#Hyperparameter Tuning with Pipeline
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

X = demo.drop('csMPa',axis=1)
y = demo['csMPa']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2, random_state=10)

pipe = Pipeline((
("sc",StandardScaler()),
("poly",PolynomialFeatures()),
("pt",PowerTransformer()),
("pca",PCA()),
("xb",XGBRegressor())
))
param_grid = {
    'poly__degree' : [1,2],
    'pca__n_components' : [30,40],
    'xb__n_estimators' : [10,20,30,40,50]
}
search = GridSearchCV(pipe,param_grid,cv=5)
search.fit(Xtrain,ytrain)

# Tried with Degree 1 - 5
# Degree = 3 gives the best result. 
model_pipe = Pipeline ((
    ("poly", PolynomialFeatures(degree=3)),
    ("lr", LinearRegression()),
    ("xg", XGBRegressor())
))

full_pipe = Pipeline((

    ("transform_pipe",transform_pipe ),
    ("model_pipe", model_pipe)
))

full_pipe.fit(Xtrain,ytrain)
