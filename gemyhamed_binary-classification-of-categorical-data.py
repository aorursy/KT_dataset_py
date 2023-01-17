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
import pandas as pd 

import numpy as np 

from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import BernoulliNB

from sklearn.impute import SimpleImputer

df_train = pd.read_csv('../input/training.csv',sep = ';',decimal=",")

pd.set_option('display.max_columns', 30)

print(df_train.head())



print(df_train.info())

df_test = pd.read_csv('../input/validation.csv',sep = ';',decimal=",")

pd.set_option('display.max_columns', 30)

print(df_test.head())

print(df_test.info())



df_train_min = df_train[df_train.columns.difference(['variable18'])]

df_train_min.dropna(how='any', inplace=True)

df_train_min.reset_index(inplace = True)

df_train_min = df_train_min[df_train_min.columns.difference(['index'])]



print(df_train_min.info())

df_test_min = df_test[df_test.columns.difference(['variable18'])]

df_test_min.dropna(how='any', inplace=True)

df_test_min.reset_index(inplace=True )

df_test_min = df_test_min[df_test_min.columns.difference(['index'])]



print(df_test_min.info())
dftm = df_train_min.append(df_test_min,ignore_index = True)

print(dftm.head())

print(dftm.info())
cat_list = dftm[['variable1','variable4','variable5','variable6','variable7','variable9','variable10',

         'variable12','variable13','classLabel']]

for i in cat_list : 

    print('For col named %s :' % i,'\n',dftm[i].value_counts())
#one hot vector

dftm_ohv = pd.get_dummies(dftm,columns=['variable1','variable4','variable5','variable6','variable7','variable9','variable10',

         'variable12','variable13','classLabel'],drop_first =True)

print(dftm_ohv.head())
dftm_ohv = dftm_ohv.astype(float)



x = dftm_ohv.drop('classLabel_yes.',axis = 1)

y = dftm_ohv['classLabel_yes.']



print(type(y))

x_train = x.iloc[0:3521,:].values

y_train = y.iloc[0:3521].values



x_test = x.iloc[3521:,:].values

y_test = y.iloc[3521:].values
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)

class_weight = dict(zip(np.unique(y_train), class_weight_list))



print(class_weight)


rfc = RandomForestClassifier(n_estimators=1,max_depth=5,class_weight=class_weight,random_state=0)



param_grid = {'n_estimators':[1,2,3],'max_depth':range(3,12)}



gs = GridSearchCV(rfc, param_grid=param_grid,scoring='f1',cv=10)

gs.fit(x_train,y_train)

y_pred = gs.predict(x_test)

print(gs.best_estimator_)

print("Best score: %0.3f" % gs.best_score_)



from sklearn.metrics import classification_report

print(gs.score(x_test,y_test))

print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import BernoulliNB

ber = BernoulliNB()

ber.fit(x_train,y_train)

y_pred=ber.predict(x_test)

print(classification_report(y_test, y_pred))
dftotal = df_train.append(df_test)

dftotal = dftotal.dropna(subset=['variable1', 'variable2','variable4','variable5','variable6','variable7','variable14','variable17']).reset_index()

print(dftotal.info())
#One Hot Vector 

dftotal = pd.get_dummies(dftotal,columns=['variable1','variable4','variable5','variable6','variable7','variable9','variable10',

         'variable12','variable13','classLabel'],drop_first =True)

dftotal['variable18'] = dftotal['variable18'].replace({'t':'1','f':'0'}).astype('category')

dftotal = dftotal[dftotal.columns.difference(['index'])]



print(dftotal.head())

print(dftotal.info())
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent")



dfemp = pd.DataFrame(imp.fit_transform(dftotal),columns =dftotal.columns )

print(dfemp.info())
dfemp_ohv = dfemp.astype(float)



x = dfemp_ohv.drop('classLabel_yes.',axis = 1)

y = dfemp_ohv['classLabel_yes.']



print(type(y))

x_train = x.iloc[0:3521,:].values

y_train = y.iloc[0:3521].values



x_test = x.iloc[3521:,:].values

y_test = y.iloc[3521:].values
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)







class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)

class_weight = dict(zip(np.unique(y_train), class_weight_list))



print(class_weight)
rfc = RandomForestClassifier(class_weight=class_weight,random_state=0)



param_grid = {'n_estimators':[1,2,3],'max_depth':range(3,12)}



gs = GridSearchCV(rfc, param_grid=param_grid,scoring='f1',cv=10)

gs.fit(x_train,y_train)

y_pred = gs.predict(x_test)

print(gs.best_estimator_)

print("Best score: %0.3f" % gs.best_score_)



from sklearn.metrics import classification_report

print(gs.score(x_test,y_test))

print(classification_report(y_test, y_pred))

ber = BernoulliNB()

ber.fit(x_train,y_train)

y_pred=ber.predict(x_test)

print(classification_report(y_test, y_pred))