import numpy as np

import pandas as pd

import sklearn

import seaborn as sns

from sklearn.metrics import accuracy_score
df = pd.read_csv("train.csv")
X = df.loc[:,['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']]

Y =  df.loc[:,['class']]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.7)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_test= scaler.fit_transform(X_test)

X_train= scaler.fit_transform(X_train)

from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_train,Y_train)

y_pred_xgb = xgb_clf.predict(X_test)

xgb_acc = accuracy_score(Y_test,y_pred_xgb)



print(xgb_acc)
test = pd.read_csv("test.csv")

test_X = test.loc[:,['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']]

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X= scaler.fit_transform(X)

test_X= scaler.transform(test_X)
# Submission 2

# xgb_clf_main = XGBClassifier().fit(X,Y)

# ypred = xgb_clf_main.predict(test_X)

# from sklearn.ensemble import RandomForestClassifier

# rand_for = RandomForestClassifier(n_estimators = 3000)

# rand_for.fit(X,Y);

# ypred = rand_for.predict(test_X)





#Submission 1

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, RandomForestClassifier



estimators = [('rf', RandomForestClassifier(n_estimators=699)), ('bag', DecisionTreeClassifier()), ('xgb', XGBClassifier(n_estimators=698))]



# soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X,Y)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X,Y)

ypred = hard_voter.predict(test_X);

ypred

y_out = [[test['id'][i],ypred[i]] for i in range(len(ypred))]

out_df = pd.DataFrame(data = y_out,columns = ['id','class'])

out_df.to_csv(r'out_13.csv',index = False)