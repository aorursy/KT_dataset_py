# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
df.head()
import seaborn as sns

sns.heatmap(df.corr())

df.corr()
from sklearn.model_selection import train_test_split



imp=df.drop('class',axis=1)



nimp=['chem_0','chem_1','chem_6','chem_4']

X_train, X_val, y_train, y_val = train_test_split(df[nimp], df['class'], test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val,y_pred)



print(accuracy)
#BaggingClassifier

from sklearn.ensemble import BaggingClassifier



bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier()).fit(X_train,y_train)

y_pred_bag = bag_clf.predict(X_val)

bag_acc = accuracy_score(y_val,y_pred_bag)



print(bag_acc)
#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



#for i in range(500):

rf_clf = RandomForestClassifier(n_estimators=250).fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_val)

rf_acc = accuracy_score(y_val,y_pred_rf)



print(rf_acc)
#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier().fit(X_train,y_train)

y_pred_gb = gb_clf.predict(X_val)

gb_acc = accuracy_score(y_val,y_pred_gb)



print(gb_acc)
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=43, solver='lbfgs',

                          multi_class='multinomial').fit(X_train, y_train)

pred=clf.predict(X_val)

acc=accuracy_score(pred,y_val);

acc
#XGBClassifier

from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_train,y_train)

y_pred_xgb = xgb_clf.predict(X_val)

xgb_acc = accuracy_score(y_val,y_pred_xgb)



print(xgb_acc)
n=len(X_train)

X_A = X_train[:n//2]

y_A = y_train[:n//2]

X_B = X_train[n//2:]

y_B = y_train[n//2:]
clf_1 = DecisionTreeClassifier().fit(X_A,y_A)

y_pred_1 = clf_1.predict(X_B)

clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A,y_A)

y_pred_2 = clf_2.predict(X_B)

clf_3 = GradientBoostingClassifier().fit(X_A,y_A)

y_pred_3 =clf_3.predict(X_B)
X_C = pd.DataFrame({'RandomForest':y_pred_2,'DecisionTrees':y_pred_1,'GradientBoost':y_pred_3})

y_C = y_B

X_C.head()
X_D = pd.DataFrame({'RandomForest':clf_2.predict(X_val),'DecisionTrees':clf_1.predict(X_val),'GradientBoost':clf_3.predict(X_val)})

y_D = y_val
from xgboost import XGBClassifier



meta_clf = XGBClassifier().fit(X_C,y_C)

y_pred_meta = meta_clf.predict(X_D)

meta_acc = accuracy_score(y_D,y_pred_meta)



print(meta_acc)
testd=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')



newtestd = pd.DataFrame({'RandomForest':clf_2.predict(testd[nimp]),'DecisionTrees':clf_1.predict(testd[nimp]),'GradientBoost':clf_3.predict(testd[nimp])})



dx = pd.DataFrame({'RandomForest':clf_2.predict(df[nimp]),'DecisionTrees':clf_1.predict(df[nimp]),'GradientBoost':clf_3.predict(df[nimp])})

dy = df['class']





nmeta_clf = XGBClassifier().fit(dx,dy)

ans=nmeta_clf.predict(newtestd)

finalans=pd.DataFrame({'id':testd["id"],'class':ans})



finalans.to_csv('submission.csv',index=False)
testd=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')



ans=rf_clf.predict(testd[nimp])

finalans=pd.DataFrame({'id':testd["id"],'class':ans})



finalans.to_csv('submissionb.csv',index=False)
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

import warnings

warnings.filterwarnings("ignore")





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
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

import warnings

warnings.filterwarnings("ignore")



parameters = {'weights':[[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (VotingClassifier(estimators=estimators, voting='soft').fit(X_train, y_train)).predict(testd[nimp])      #Using the unoptimized classifiers, generate predictions

optimized_predictionsa = best_clf_sv.predict(testd[nimp])        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_op))
testd=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')



ans=rf_clf.predict(testd[nimp])

finalansa=pd.DataFrame({'id':testd["id"],'class':optimized_predictionsa})



finalansa.to_csv('submissiona.csv',index=False)