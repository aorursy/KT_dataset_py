import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb

import os
print(os.listdir("../input"))


col_list = ['Pelvic_incidence',
               'Pelvic_tilt',
               'Lumbar_lordosis_angle',
               'Sacral_slope',
               'Pelvic_radius',
               'Degree_spondylolisthesis',
               'Pelvic_slope',
               'Direct_tilt',
               'Thoracic_slope',
               'Cervical_tilt',
               'Sacrum_angle',
               'Scoliosis_slope',
               'Attribute',
               'To_drop']
df = pd.read_csv("../input/Dataset_spine.csv",names = col_list,header=1)
df.head()
sns.set_style("white")
g=sns.factorplot(x='Attribute', hue='Attribute', data= df, kind='count',size=5,aspect=.8)
df.drop('To_drop',axis=1,inplace = True)
df['Attribute'] = df['Attribute'].map({'Abnormal':1,'Normal':0})

df = shuffle(df)

df.info()
X = np.array(df.ix[:, df.columns != 'Attribute'])
y = np.array(df.ix[:, df.columns == 'Attribute'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
def XGB_param_selection(X, y, nfolds):
    learning_rate = [0.001, 0.01, 0.1, 1]
    nos = np.linspace(1, 200, num=200)
    depth = np.linspace(1, 20, num=20)
    numbers = [ int(x) for x in nos ]
    m_d = [ int(x) for x in depth ]
    param_grid = {'n_estimators': numbers,'max_depth':m_d,'learning_rate':learning_rate}
    grid_search = GridSearchCV(xgb.XGBClassifier(subsample=0.5), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
print(XGB_param_selection(X_train_res,y_train_res,5))
clf = xgb.XGBClassifier(n_estimators=192, max_depth=5, learning_rate=0.1, subsample=0.5)
clf.fit(X_train_res, y_train_res)
predicted = clf.predict(X_test)
print(accuracy_score(y_test, predicted))