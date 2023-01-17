# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import sklearn

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.metrics import precision_recall_fscore_support as score



from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



from time import time



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, RANSACRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.linear_model import Lasso, Ridge





from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR





import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn as sb
hr = pd.read_csv("../input/core_dataset.csv")

hr
hr.columns = ['Name','EmpNum','State','Zip','Date_of_birth','Age','Sex','MaritalDesc','CitizenDesc','Hispanic/Latino','RaceDesc','Date_of_Hire','Date_of_Termination','Reason_For_Termination','Employment_Status','Department','Position','Pay_Rate','Manager_Name','Employment_Source','Performance_Score']

hr = hr.drop(hr.index[301])

print(hr)
hr_no_dot = hr.drop('Date_of_Termination',axis=1)
le = preprocessing.LabelEncoder()

hr_no_dot = hr_no_dot.apply(le.fit_transform)

hr_no_dot
corr = hr_no_dot.corr()

print(corr)
hr_no_dot.columns
X = hr_no_dot.loc[:,['State', 'Zip', 'Date_of_birth', 'Age', 'Sex',

       'MaritalDesc', 'CitizenDesc', 'Hispanic/Latino', 'RaceDesc',

       'Date_of_Hire', 'Reason_For_Termination', 'Employment_Status',

       'Department', 'Position', 'Manager_Name',

       'Employment_Source', 'Performance_Score']]

y = hr_no_dot.loc[:,['Pay_Rate']]



print(X)

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=17)
regressors = [

    LinearRegression(), 

    RANSACRegressor(), 

    KNeighborsRegressor(),

    KNeighborsRegressor(n_neighbors=9, metric='manhattan'),

    SVR(),

    LinearSVR(),

    GaussianProcessRegressor(),

    SVR(kernel='linear'), # Cf. LinearSVR: much slower, might be better or worse: 

]
head = 6

for model in regressors[:head]:

    start = time()

    model.fit(X_train, y_train)

    train_time = time() - start

    start = time()

    predictions = model.predict(X_test)

    predict_time = time()-start    

    print(model)

    print("\tTraining time: %0.3fs)" % train_time)

    print("\tPrediction time: %0.3fs" % predict_time)

    print("\tExplained variance:", explained_variance_score(y_test, predictions))

    print("\tMean absolute error:", mean_absolute_error(y_test, predictions))

    print("\tR2 score:", r2_score(y_test, predictions))

    print()
lr = LinearRegression()

lasso = Lasso()

ridge = Ridge()



for model in [lr, lasso, ridge]:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(model)

    print("\tExplained variance:", explained_variance_score(y_test, predictions))

    print("\tMean absolute error:", mean_absolute_error(y_test, predictions))

    print("\tR2 score:", r2_score(y_test, predictions))

    print()
regressors = [LinearRegression(),

              DecisionTreeRegressor(max_depth=5),

              DecisionTreeRegressor(max_depth=10),

              DecisionTreeRegressor(max_depth=20),

              RandomForestRegressor(max_depth=10),

              SVR(),]



for model in regressors:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(model)

    print("\tExplained variance:", explained_variance_score(y_test, predictions))

    print("\tMean absolute error:", mean_absolute_error(y_test, predictions))

    print("\tR2 score:", r2_score(y_test, predictions))

    print()