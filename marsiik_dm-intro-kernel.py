import numpy as np

import pandas as pd



import sklearn.externals.joblib as jl



import sklearn.cross_validation as cv

import sklearn.feature_extraction as fe



import sklearn.svm as svm



import matplotlib.pyplot as plt



import sklearn.linear_model as lm



import sklearn.preprocessing as preprocessing



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
train_df = pd.read_csv('../input/train.csv', low_memory=False)

test_df = pd.read_csv('../input/test.csv')

store_df = pd.read_csv('../input/store.csv')
train_df = train_df[train_df.Open != 0]

test_df.fillna(1, inplace=True)
train_df = pd.merge(train_df, store_df, on='Store')

test_df = pd.merge(test_df, store_df, on='Store')



sale_means = train_df.groupby('Store').mean().Sales

sale_means.name = 'Sales_Means'



train_df = train_df.join(sale_means,on='Store')

test_df = test_df.join(sale_means,on='Store')
train_df.head()
y_train = train_df.Sales.tolist()



train_df_d = train_df.drop(['Date','Sales','Store','Customers'],axis=1).fillna(0)



train_df_dic = train_df_d.fillna(0).to_dict('records')



test_df_dic = test_df.drop(["Date","Store","Id"],axis=1).fillna(0).to_dict('records')
dict_vect = fe.DictVectorizer()

X_train = dict_vect.fit_transform(train_df_dic)

X_test = dict_vect.transform(test_df_dic)
maxmin = preprocessing.MinMaxScaler()

X_train = maxmin.fit_transform(X_train.toarray())

X_test = maxmin.transform(X_test.toarray())
Xtrain, Xtest, Ytrain, Ytest = cv.train_test_split(X_train,y_train)
clf = RandomForestRegressor()

parameters = {'n_estimators':range (1,20), 'min_samples_leaf':range (1, 10), 'max_depth':range(1, 15)}

grid_search = GridSearchCV (clf, parameters, verbose=1, n_jobs=12)

grid_search.fit (Xtrain, Ytrain)

print (grid_search.best_params_)
clf = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, max_depth=2, criterion='mse')

clf.fit(Xtrain,Ytrain)
q = [i for i in zip(dict_vect.feature_names_,clf.feature_importances_) ]



q = pd.DataFrame(q, columns = ['Feature_Names','Importance'],index=dict_vect.feature_names_)



q_chart = q.sort('Importance').plot(kind='barh',layout='Feature_Names')



fig_q = q_chart.get_figure()

fig_q.savefig('feature_impartance.png')
predicted = clf.predict(Xtest)

predicted = np.array(predicted)



Ytest = np.array(Ytest)



np.abs((predicted - Ytest)).sum() / len(predicted)
pred = clf.predict(X_test)

output = pd.DataFrame(test_df.Id).join(pd.DataFrame(pred, columns=['Sales']))

output.to_csv('first_attempt.csv',index=False)
import numpy as np

import pandas as pd



import sklearn.externals.joblib as jl



import sklearn.cross_validation as cv

import sklearn.feature_extraction as fe



import sklearn.svm as svm



import matplotlib.pyplot as plt



import sklearn.linear_model as lm



import sklearn.preprocessing as preprocessing



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
train_df = pd.read_csv('../input/train.csv', low_memory=False)

test_df = pd.read_csv('../input/test.csv')

store_df = pd.read_csv('../input/store.csv')
train_df = train_df[train_df.Open != 0]

test_df.fillna(1, inplace=True)
train_df = pd.merge(train_df, store_df, on='Store')

test_df = pd.merge(test_df, store_df, on='Store')



sale_means = train_df.groupby('Store').mean().Sales

sale_means.name = 'Sales_Means'



train_df = train_df.join(sale_means,on='Store')

test_df = test_df.join(sale_means,on='Store')
train_df.head()
y_train = train_df.Sales.tolist()



train_df_d = train_df.drop(['Date','Sales','Store','Customers'],axis=1).fillna(0)



train_df_dic = train_df_d.fillna(0).to_dict('records')



test_df_dic = test_df.drop(["Date","Store","Id"],axis=1).fillna(0).to_dict('records')
dict_vect = fe.DictVectorizer()

X_train = dict_vect.fit_transform(train_df_dic)

X_test = dict_vect.transform(test_df_dic)
maxmin = preprocessing.MinMaxScaler()

X_train = maxmin.fit_transform(X_train.toarray())

X_test = maxmin.transform(X_test.toarray())
Xtrain, Xtest, Ytrain, Ytest = cv.train_test_split(X_train,y_train)

clf = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, max_depth=2, criterion='mse', verbose=1)

clf.fit(Xtrain,Ytrain)
q = [i for i in zip(dict_vect.feature_names_,clf.feature_importances_) ]



q = pd.DataFrame(q, columns = ['Feature_Names','Importance'],index=dict_vect.feature_names_)



q_chart = q.sort('Importance').plot(kind='barh',layout='Feature_Names')



fig_q = q_chart.get_figure()

fig_q.savefig('feature_impartance.png')
predicted = clf.predict(Xtest)

predicted = np.array(predicted)



Ytest = np.array(Ytest)



np.abs((predicted - Ytest)).sum() / len(predicted)
pred = clf.predict(X_test)

output = pd.DataFrame(test_df.Id).join(pd.DataFrame(pred, columns=['Sales']))

output.to_csv('first_attempt.csv',index=False)
