#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection



sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)



#%reset -f

dataset = pd.read_csv('../input/Concrete_Data_Yeh.csv')
print(dataset.columns[dataset.isnull().any()])

print(dataset.isnull().sum())



#Duplicate check

print(sum(dataset.duplicated(dataset.columns)))

dataset = dataset.drop_duplicates(dataset.columns, keep='last')

import phik

from phik import resources, report



corr = dataset.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

abs(dataset.skew())

dataset['age'] = np.log(dataset['age'])

#X= dataset.drop('csMPa', axis=1)

#y= dataset['csMPa']

#

#X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)

#

#from sklearn.feature_selection import RFECV

#from sklearn.ensemble import RandomForestRegressor

#

## The "accuracy" scoring is proportional to the number of correct classifications

#clf_rf_4 = RandomForestRegressor() 

#rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='neg_mean_squared_error')   #5-fold cross-validation

#rfecv = rfecv.fit(X_train, y_train)

#

#print('Optimal number of features :', rfecv.n_features_)

#print('Best features :', X_train.columns[rfecv.support_])

#

#

#cl = [col for col in X_train.columns if col not in X_train.columns[rfecv.support_]]

#

X= dataset.drop(['csMPa','superplasticizer'], axis=1) 

y= dataset['csMPa']

#Splitting the Dataset into Training set and Test Set

train_x,test_x,train_y,test_y = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)





#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

train_x = sc.fit_transform(train_x)

test_x = sc.transform(test_x)



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.model_selection import cross_val_score



clf_lr = LinearRegression()

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred = clf_lr.predict(test_x)

print('')

print('####### Linear Regression #######')

print('Score : %.4f' % clf_lr.score(test_x, test_y))

print(accuracies)



mse = mean_squared_error(test_y, y_pred)

mae = mean_absolute_error(test_y, y_pred)

rmse = mean_squared_error(test_y, y_pred)**0.5

r2 = r2_score(test_y, y_pred)



print('')

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)





from sklearn.ensemble import RandomForestRegressor

#from sklearn.model_selection import GridSearchCV

#params = [{'n_estimators':[100,15], 'max_depth':[4,None]}]

#vc = GridSearchCV(estimator=RandomForestRegressor(),param_grid=params,

#                  verbose=1,cv=10,n_jobs=-1)

#res = vc.fit(train_x,train_y)

#print(res.best_score_)

#print(res.best_params_)



clf_lr = RandomForestRegressor()

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred = clf_lr.predict(test_x)

print('')

print('####### Forest #######')

print('Score : %.4f' % clf_lr.score(test_x, test_y))

print(accuracies)



mse = mean_squared_error(test_y, y_pred)

mae = mean_absolute_error(test_y, y_pred)

rmse = mean_squared_error(test_y, y_pred)**0.5

r2 = r2_score(test_y, y_pred)



print('')

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)
