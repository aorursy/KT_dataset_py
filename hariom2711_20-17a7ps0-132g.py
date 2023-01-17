import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import  metrics 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

X = df.drop('class',axis=1)

y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)



target = 'class'

IDcol = 'id'
xgb1 = xgb.sklearn.XGBClassifier(

 learning_rate =0.05,

 objective='multi:softmax',

 scale_pos_weight=1,

 seed=27,

 colsample_bytree= 0.6, 

 subsample=0.8,

 n_estimators=100,

 min_child_weight= 1,

 gamma=0.4,

 max_depth= 5,

 reg_alpha=1e-05)    

ab = xgb1.fit(X_train,y_train)

f_val = ab.predict(X_val)

metrics.accuracy_score(f_val, y_val)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train,y_train)

y_pred = clf.predict(X_val)

accuracy = metrics.accuracy_score(y_val,y_pred)



print(accuracy)
from sklearn.ensemble import AdaBoostClassifier

# ad_clf = AdaBoostClassifier(n_estimators=100)

ad_clf = AdaBoostClassifier(n_estimators=100).fit(X_train,y_train)

y_pred = clf.predict(X_val)

accuracy = metrics.accuracy_score(y_val,y_pred)

accuracy
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import  accuracy_score

bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier()).fit(X_train,y_train)

y_pred_bag = bag_clf.predict(X_val)

bag_acc = accuracy_score(y_val,y_pred_bag)



print(bag_acc)
from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier(n_estimators=1000).fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_val)

rf_acc = accuracy_score(y_val,y_pred_rf)



print(rf_acc)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1500).fit(X_train,y_train)

y_pred_gb = gb_clf.predict(X_val)

gb_acc = accuracy_score(y_val,y_pred_gb)



print(gb_acc)
from sklearn.ensemble import VotingClassifier



estimators = [('rf', rf_clf), ('gbc', gb_clf), ('xgb', xgb1), ('bag', bag_clf)]



soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)



soft_acc = accuracy_score(y_val,soft_voter.predict(X_val))

hard_acc = accuracy_score(y_val,hard_voter.predict(X_val))



print("Acc of soft voting classifier:{}".format(soft_acc))

print("Acc of hard voting classifier:{}".format(hard_acc))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



parameters = {'weights':[[2,1,2,1],[2,2,2,1],[3,2,3,2],[2,2,3,2],[3,1,3,1],[3,2,3,1],[2,2,2,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (VotingClassifier(estimators=estimators, voting='soft').fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf_sv.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
grid_obj.best_params_
n = len(X_train)

X_A = X_train[:n//2]

y_A = y_train[:n//2]

X_B = X_train[n//2:]

y_B = y_train[n//2:]
clf_1 = gb_clf.fit(X_A, y_A)

y_pred_1 = clf_1.predict(X_B)

clf_2 = rf_clf.fit(X_A, y_A)

y_pred_2 = clf_2.predict(X_B)

clf_3 = xgb1.fit(X_A, y_A)

y_pred_3 = clf_3.predict(X_B)

clf_4 = bag_clf.fit(X_A, y_A)

y_pred_4 = clf_4.predict(X_B)

X_C = pd.DataFrame({'RandomForest': y_pred_2, 'gb': y_pred_1, 'xgb': y_pred_3, 'bag': y_pred_4})

y_C = y_B

X_C.head()
X_D = pd.DataFrame({'RandomForest': clf_2.predict(X_val), 'gb': clf_1.predict(X_val), 'xgb': clf_3.predict(X_val), 'bag': clf_4.predict(X_val)})

y_D = y_val
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {}

params['learning_rate'] = 0.001

params['boosting_type'] = 'goss'

params['objective'] = 'multiclass'

params['metric'] = 'multi_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 10000

params['num_iterations'] = 1000 

params['min_data'] = 5

params['max_depth'] = 10

params['num_class'] = 7+1

params['lambda_l1'] = 1e-5



lgb_clf = lgb.train(params, d_train, 100)



lgb_pred=clf.predict(X_val)



lgb_accuracy=accuracy_score(lgb_pred,y_val)

lgb_accuracy
lgb_pred=clf.predict(X_val)
lgb_accuracy=accuracy_score(lgb_pred,y_val)

lgb_accuracy
fin1 = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

pdr = best_clf_sv.predict(fin1)

y_pd=pd.Series(pdr)

y_pd

y_pd1=fin1['id']

ans = pd.DataFrame()

ans['id'] = y_pd1

ans['class'] = y_pd

ans.to_csv('soln.csv', index=False)
