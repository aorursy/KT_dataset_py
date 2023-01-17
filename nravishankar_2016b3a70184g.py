# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure()

sns.heatmap(df.corr())
# from google.colab import files 

# uploaded = files.upload()
#df = pd.read_csv('/content/train.csv')

df.head()

df = df.fillna(df.mean())



t = []

t = pd.get_dummies(df['type'])

df['type2'] = t['new']

df['type2-2'] = t['old']



indep = ['id', 'feature1', 'feature2','feature3', 'feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11', 'type2', 'type2-2']
X = df[indep]

y = df.iloc[:, 13].values
X.head()

#y[0:10]

#preprocessing.normalize(X)
# from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler



# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)

# from sklearn.preprocessing import RobustScaler



# scalar = StandardScaler()

# X_train = scalar.fit_transform(X_train)

# X_test = scalar.transform(X_test)



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn import metrics



from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=700, max_features = 'sqrt', bootstrap=True)





'''scorer = make_scorer(metrics.mean_squared_error)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_ '''

rfc.fit(X,y)

#y_pred=rfc.predict(X)
from sklearn.ensemble import ExtraTreesRegressor



modelETR =  ExtraTreesRegressor(n_estimators = 2000,  n_jobs = -1)



modelETR.fit(X, y)



# y_pred2 = modelETC.predict(X_test)

# y_pred2 = np.round(y_pred2)







from sklearn import metrics





#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))



# n_estimators = [100, 300, 500]

# max_depth = [5, 8, 15, 25, 30]

# min_samples_split = [2, 5, 10, 15]

# min_samples_leaf = [1, 2, 5, 10] 



# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  

#               min_samples_split = min_samples_split, 

#              min_samples_leaf = min_samples_leaf)



# gridF = GridSearchCV(rfc, hyperF, cv = 3, verbose = 1, 

#                       n_jobs = -1)

# bestF = gridF.fit(X_train, y_train)
df_test=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df_test = df_test.fillna(df_test.mean())

t = pd.get_dummies(df_test['type'])

df_test['type2'] = t['new']

df_test['type2-2'] = t['old']

test_indep = df_test[indep]





#test_indep = scalar.transform(test_indep)

#norm_test_indep = preprocessing.normalize(test_indep)

#norm_test_indep = sc.fit_transform(norm_test_indep)



y_ans = modelETR.predict(test_indep)



y_ans = np.round(y_ans)





#y_ans = rfc.predict(test_indep)

y_ans[0]

finalans=pd.DataFrame({'id':df_test["id"],'rating':y_ans})



#export_csv = df.to_csv (r'/home/submission.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

finalans.to_csv('submission7.csv', index=False)
