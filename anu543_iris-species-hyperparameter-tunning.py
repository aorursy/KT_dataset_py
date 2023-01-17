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
import os 
os.listdir('../input')
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/iris/Iris.csv')

print("Size   of   Dataset   :- " , df.shape)
print("Features of Dataset   :-" , df.columns)
print("\n\n Missing Value:-"  , df.isna().sum())
print("\n\n unique value =        :- ", df.nunique())
df.head()
sns.scatterplot(df['SepalLengthCm'] , df['SepalWidthCm'] , color = 'red')
sns.scatterplot(df['PetalWidthCm'] , df['PetalWidthCm'] , color = 'green')
# Visualization for Numerical variable 
fig ,axs  = plt.subplots(1,4 ,figsize = (20,4))
for i ,f in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
    sns.distplot(df[f] , ax= axs[i] , kde=True)
df.dtypes
# Target Value
print(df['Species'].unique())
sns.countplot(df['Species'] , label = 'count');
x= df.drop('Species' , axis=1)
y=df['Species']
x.shape , y.shape
from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(x , y ,test_size = .20 , random_state = 101)
x_train.shape , y_train.shape , x_test.shape ,y_test.shape
# This dataset is based on categorical target..so we need to apply the CLassfication algo.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_model = LogisticRegression()
log_model.fit(x_train ,y_train)
pred1 = log_model.predict(x_test)
pred1
print("Accuracy for Logistic Model = " , accuracy_score(y_test , pred1))

from sklearn.svm import SVC
model = SVC(kernel='linear' ,random_state =1)
model.fit(x_train ,y_train)
pred = model.predict(x_test)
pred
model.score(x_test ,y_test)
# Creating Confusion matrix
from sklearn.metrics import confusion_matrix ,accuracy_score
confusion_matrix(y_test ,pred)
accu = accuracy_score(y_test ,pred)
print("Accuracy Score = ",accu)
# with Hyperparameter for SVM
from sklearn.model_selection import GridSearchCV
parameters =  {'C':[1,10,100,1000] ,'kernel':['linear'] , 'gamma' :[0.1 , 0.2 ,0.3 ,0.4 ,0.5]}
grid_search = GridSearchCV(estimator= model ,param_grid=parameters ,scoring=accu ,cv=10 )

# with hyperparameter for Radomforest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

parametre_grid={'n_estimators':[5,10,25,50,100],
               'criterion':['gini','entropy'],
               'max_features':[1,2,3,4],
               'warm_start':[True,False]}

#crs_validation= StratifiedKFold(n_splits=10, shuffle=True ,random_state=101)

grid_search=GridSearchCV(rf,parametre_grid ,cv=10 )

grid = grid_search.fit(x_train.values,y_train.values)
grid
print('Best score{}'.format(grid.best_score_))
print('Best Parametrs{}'.format(grid.best_params_))