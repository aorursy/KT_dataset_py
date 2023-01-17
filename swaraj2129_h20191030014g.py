import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

#df.info()
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.info()
df
(df.corr())
cols_to_drop = ['id']

df = df.drop(cols_to_drop, axis=1)

df.head()
missing_count = df.isnull().sum()

missing_count[missing_count > 0]

df[df.chem_6 == 0.0]
df['chem_2'].replace(0, np.nan, inplace = True)

df['chem_2'].replace(0, np.nan, inplace = True)
df["chem_1"]=df["chem_1"].fillna(df["chem_1"].mean())

df["chem_2"]=df["chem_2"].fillna(df["chem_2"].mean())
X = df.drop('class',axis=1)

y = df['class']



print (X.shape, y.shape)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessRegressor

from xgboost import XGBClassifier
# reg1 = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1]).fit(X_train,y_train)

# reg2 = LinearRegression().fit(X_train,y_train)

# reg3 = Ridge().fit(X_train,y_train)

# reg4 = Lasso().fit(X_train,y_train)

# reg5 = ElasticNet().fit(X_train,y_train)

# reg6 = BayesianRidge().fit(X_train,y_train)

# reg7 = KNeighborsRegressor().fit(X_train,y_train)

# reg8 = DecisionTreeRegressor().fit(X_train,y_train)

#reg9 = GradientBoostingClassifier().fit(X_train,y_train)

# reg10 = GradientBoostingRegressor().fit(X_train,y_train)

# reg11 = AdaBoostRegressor().fit(X_train,y_train)

# reg12 = LogisticRegression().fit(X_train,y_train)

# reg13 = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)

# reg14 = RandomForestRegressor(n_estimators=500).fit(X_train,y_train)

#reg15 = MLPClassifier().fit(X_train,y_train)

# reg16 = MLPRegressor().fit(X_train,y_train)

#reg17 = ExtraTreesRegressor(n_estimators=500, max_depth=50).fit(X_train,y_train)

#reg18 = ExtraTreesClassifier(n_estimators=900,min_samples_leaf=1,max_depth=None).fit(X_train,y_train)

# reg19 = GaussianNB().fit(X_train,y_train)

# reg20 = GaussianProcessRegressor.fit(X_train,y_train)

reg21 = XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.1, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=100, n_jobs=1,

                                     nthread=None, objective='binary:logistic',

                                     random_state=0, reg_alpha=0,

                                     scale_pos_weight=1, seed=None, silent=None,

                                     subsample=1, verbosity=1).fit(X_train, y_train)
# from sklearn.metrics import accuracy_score

# y_pred = reg9.predict(X_val)

# accuracy = accuracy_score(y_val,y_pred)



# print(accuracy)
# from sklearn.metrics import accuracy_score

# y_pred = reg18.predict(X_val)

# accuracy = accuracy_score(y_val,y_pred)



# print(accuracy)
from sklearn.metrics import accuracy_score

y_pred = reg21.predict(X_val)

print(accuracy_score(y_val,y_pred))
# from sklearn.model_selection import GridSearchCV

 

# param_grid = {

#     'min_samples_split':[4,2,3],

#     'min_samples_leaf': [1,2,3],

#     'max_depth': [50,100,None],

#     'n_estimators': [500]

# }



# rf = ExtraTreesClassifier()



# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
# grid_search.best_params_
# from sklearn.ensemble import VotingClassifier

# from sklearn.ensemble import BaggingClassifier

# from xgboost import XGBClassifier



# estimators = [('xgb', XGBClassifier(base_score=0.5, booster='gbtree',

#                                      colsample_bylevel=1, colsample_bynode=1,

#                                      colsample_bytree=1, gamma=0,

#                                      learning_rate=0.1, max_delta_step=0,

#                                      max_depth=3, min_child_weight=1,

#                                      missing=None, n_estimators=100, n_jobs=1,

#                                      nthread=None, objective='binary:logistic',

#                                      random_state=0, reg_alpha=0,

#                                      scale_pos_weight=1, seed=None, silent=None,

#                                      subsample=1, verbosity=1)),('rf',RandomForestClassifier(n_estimators=500)),('ab',AdaBoostClassifier())]



# soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

# hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)
# soft_acc = accuracy_score(y_val,soft_voter.predict(X_val))

# hard_acc = accuracy_score(y_val,hard_voter.predict(X_val))



# print("Acc of soft voting classifier:{}".format(soft_acc))

# print("Acc of hard voting classifier:{}".format(hard_acc))
test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

y1 = test.drop(['id'],axis = 1)

y1.head(3)

y2 = test.drop(['id'],axis = 1)
y1 = scaler.transform(y1)
y2.head()
pred =  reg21.predict(y1)
#y1.head()

cols_to_drop = ['chem_0','chem_1','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

y2 = y2.drop(cols_to_drop, axis=1)
y2.head()
y2['id'] = test['id']

y2['class'] = pred

y2

#e = pd.read_csv('ans1.csv')



#accuracy = accuracy_score(pred,e['class'])

#print(accuracy)
y2.to_csv("ans1.csv",index=False)