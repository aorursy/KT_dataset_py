import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')
df.head()
df.info()
df.describe()
missing_count = df.isnull().sum()

missing_count[missing_count > 0]
df.corr()
y=df['class'].copy()
y
y.value_counts()
df.drop(['id','chem_2','chem_3','chem_5','chem_7','attribute','class'],axis=1,inplace=True)
df.head()
numerical_features=['chem_0','chem_1','chem_4','chem_6']

X=df[numerical_features]
X.head()
y
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val,y_pred)



print(accuracy)
from sklearn.ensemble import BaggingClassifier



bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier()).fit(X_train,y_train)

y_pred_bag = bag_clf.predict(X_val)

bag_acc = accuracy_score(y_val,y_pred_bag)



print(bag_acc)
from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier().fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_val)

rf_acc = accuracy_score(y_val,y_pred_rf)



print(rf_acc)
from sklearn.ensemble import AdaBoostClassifier



ab_clf = AdaBoostClassifier().fit(X_train,y_train)

y_pred_ab = ab_clf.predict(X_val)

ab_acc = accuracy_score(y_val,y_pred_ab)



print(ab_acc)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier().fit(X_train,y_train)

y_pred_gb = gb_clf.predict(X_val)

gb_acc = accuracy_score(y_val,y_pred_gb)



print(gb_acc)
from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_train,y_train)

y_pred_xgb = xgb_clf.predict(X_val)

xgb_acc = accuracy_score(y_val,y_pred_xgb)



print(xgb_acc)
n = len(X_train)

X_A = X_train[:n//2]

y_A = y_train[:n//2]

X_B = X_train[n//2:]

y_B = y_train[n//2:]
clf_1 = DecisionTreeClassifier().fit(X_A, y_A)

y_pred_1 = clf_1.predict(X_B)

clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A, y_A)

y_pred_2 = clf_2.predict(X_B)

clf_3 = GradientBoostingClassifier().fit(X_A, y_A)

y_pred_3 = clf_3.predict(X_B)
X_C = pd.DataFrame({'RandomForest': y_pred_2, 'DeccisionTrees': y_pred_1, 'GradientBoost': y_pred_3})

y_C = y_B

X_C.head()
X_D = pd.DataFrame({'RandomForest': clf_2.predict(X_val), 'DeccisionTrees': clf_1.predict(X_val), 'GradientBoost': clf_3.predict(X_val)})

y_D = y_val
from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_C,y_C)

y_pred_xgb = xgb_clf.predict(X_D)

xgb_acc = accuracy_score(y_D,y_pred_xgb)



print(xgb_acc)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf1 = GradientBoostingClassifier().fit(X_C,y_C)

y_pred_gb1 = gb_clf1.predict(X_D)

gb_acc1 = accuracy_score(y_D,y_pred_gb1)



print(gb_acc1)
from sklearn.ensemble import VotingClassifier



estimators = [('rf', RandomForestClassifier()), ('bag', BaggingClassifier()), ('xgb', XGBClassifier())]



soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)
soft_acc = accuracy_score(y_val,soft_voter.predict(X_val))

hard_acc = accuracy_score(y_val,hard_voter.predict(X_val))



print("Acc of soft voting classifier:{}".format(soft_acc))

print("Acc of hard voting classifier:{}".format(hard_acc))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



parameters = {'weights':[[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]}    #Dictionary of parameters

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
df2=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
df2.head()
df2.info()
missing_count = df2.isnull().sum()

missing_count[missing_count > 0]
df2.drop(['id','chem_2','chem_3','chem_5','chem_7','attribute'],axis=1,inplace=True)
df2.head()
numerical_features1=['chem_0','chem_1','chem_4','chem_6']

X1=df2[numerical_features1]
X1.head()
X_E = pd.DataFrame({'RandomForest': clf_2.predict(X1), 'DeccisionTrees': clf_1.predict(X1), 'GradientBoost': clf_3.predict(X1)})

cc1= xgb_clf.predict(X_E)
df3=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df3.head()
df4=pd.DataFrame(index=df3['id'])

df4['class']=cc1
df4['class'].value_counts()
y.value_counts()
cc2=soft_voter.predict(X1)
df5=pd.DataFrame(index=df3['id'])

df5['class']=cc2
df5['class'].value_counts()
df5.to_csv('sub1.csv')
#for 2 best solution-
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(bootstrap=True,max_features='auto',max_depth=16, random_state=0, n_estimators=200,warm_start=True).fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_val)

rf_acc = accuracy_score(y_val,y_pred_rf)



print(rf_acc)
rff=rf_clf.predict(X1)
df3=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df3.head()
df4=pd.DataFrame(index=df3['id'])

df4['class']=rff
df4['class'].value_counts()
df4.to_csv('sub2.csv')