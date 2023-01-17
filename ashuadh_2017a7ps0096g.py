import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test_data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df_submit = pd.DataFrame()

df_submit['id'] =  test_data['id']
df = train_data.drop(columns = 'id')

test = test_data.drop(columns = 'id')
df.head()
df.isnull().sum()
df.fillna(df.mean(), inplace = True)
test.fillna(-1, inplace = True)
temp_code = {'new':0, 'old': 1}

df['type'] = df['type'].map(temp_code)



#One-hot

df = pd.get_dummies(data=df,columns=['type'])





test['type'] = test['type'].map(temp_code)



#One-hot

test = pd.get_dummies(data=test,columns=['type'])
df.hist(bins = 50, figsize = (12, 8))

#therer are outliers in the data but removing them didn't increase the accuracy
X = df.drop(columns = 'rating')

y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([

    ('normalizer', StandardScaler()), #Step1 - normalize data

    ('clf', LogisticRegression()) #step2 - classifier

])
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



clfs = []

clfs.append(LogisticRegression())

clfs.append(SVC())

clfs.append(SVC())

clfs.append(KNeighborsClassifier(n_neighbors=3))

clfs.append(DecisionTreeClassifier())

clfs.append(RandomForestClassifier())

clfs.append(GradientBoostingClassifier())



for classifier in clfs:

    pipeline.set_params(clf = classifier)

    scores = cross_validate(pipeline, X_train, y_train)

    print('---------------------------------')

    print(str(classifier))

    print('-----------------------------------')

    for key, values in scores.items():

            print(key,' mean ', values.mean())

            print(key,' std ', values.std())
#Random forest gives best score out of these classifiers

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score  #Find out what is accuracy_score
rfc = RandomForestClassifier()

parameters = {'n_estimators':[50, 500, 5000]}    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rfc,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train





print(grid_fit.best_params_)
#n_estimators = 5000 is the best parameter

rfc = RandomForestClassifier(n_estimators = 5000, random_state = 30, oob_score=True)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
rfc.fit(X, y)
df_submit['rating'] = rfc.predict(test)
df_submit.to_csv('Final1.csv', index = False)