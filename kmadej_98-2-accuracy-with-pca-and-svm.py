import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/data.csv')



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



y = data.diagnosis   

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#normalization

x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())

x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())



from sklearn.decomposition import PCA, NMF

from sklearn import svm

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.feature_selection import SelectKBest, chi2



from sklearn.model_selection import GridSearchCV

    

pca = PCA()

svc = svm.SVC(kernel='linear')



model = Pipeline([

    ('reduce_dim', PCA()),

    ('svc', svc)

])



# for tuning parameter





param_grid = [

  {

    'reduce_dim': [PCA()],

    'svc__C': [0.1, 5, 1, 10], 

    'reduce_dim__n_components': [3,4,5,6],

    'svc__kernel': ['linear']

  }

]

    

clf = GridSearchCV(model,param_grid,cv=3,scoring="accuracy")



clf.fit(x_train_N, y_train)



print("The best parameter found on development set is :")

print(clf.best_params_)



print("The best score is ")

print(clf.best_score_)



y_pred = clf.predict(x_test_N)



print('Accuracy is: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True,fmt="d")