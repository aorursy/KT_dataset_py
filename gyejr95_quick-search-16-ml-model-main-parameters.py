from sklearn.datasets import load_boston

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
boston = load_boston()

cancer = load_breast_cancer()



bos_df = pd.DataFrame(boston['data'],columns = boston['feature_names'])

bos_df['target'] = boston['target']



can_df = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])

can_df['target'] = cancer['target']
def train_test_clf(variable, target, test_size):

    X_train, X_test, y_train, y_test = train_test_split(variable, target, stratify = target, test_size = test_size, random_state = 0)

    

    return X_train, X_test, y_train, y_test



def train_test_reg(variable, target, test_size):

    X_train, X_test, y_train, y_test = train_test_split(variable, target, test_size = test_size, random_state = 0)

    

    return X_train, X_test, y_train, y_test
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = KNeighborsRegressor()



reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)



clf = KNeighborsClassifier()



clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.svm import SVR
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = SVR()



reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)



clf = SVC()



clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = LinearRegression()



reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = Ridge()



reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = Lasso()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.linear_model import ElasticNet
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = ElasticNet()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = LogisticRegression()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = GaussianNB()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.ensemble import GradientBoostingRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = GradientBoostingRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = GradientBoostingClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.ensemble import AdaBoostRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = AdaBoostRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.ensemble import AdaBoostClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = AdaBoostClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = xgb.XGBRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = xgb.XGBClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
import lightgbm as lgb
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = lgb.LGBMRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = lgb.LGBMClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
import catboost as cb
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = cb.CatBoostRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = cb.CatBoostClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = DecisionTreeRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = DecisionTreeClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.ensemble import ExtraTreesRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = ExtraTreesRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.ensemble import ExtraTreesClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = ExtraTreesClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_reg(np.array(bos_df.drop(columns = 'target')), np.array(bos_df['target']), 0.25)





reg = RandomForestRegressor()

   

reg.fit(X_train, y_train)
print(f'Training set score : {reg.score(X_train, y_train)}')

print(f'Test set score : {reg.score(X_test,y_test)}')
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_clf(np.array(can_df.drop(columns = 'target')), np.array(can_df['target']), 0.25)





clf = RandomForestClassifier()

   

clf.fit(X_train, y_train)
print(f'Training set score : {clf.score(X_train, y_train)}')

print(f'Test set score : {clf.score(X_test,y_test)}')