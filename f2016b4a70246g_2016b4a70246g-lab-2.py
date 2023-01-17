import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
df=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')
df.head()
df.isna().sum()
sns.heatmap(df.corr())
X = df.loc[:,['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']]

y_1 = df.loc[:,['class']]

y = np.array(y_1)

test_X = test.loc[:,['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']]
estimators = [('rf',RandomForestClassifier()),('dt', DecisionTreeClassifier()), ('xgb', XGBClassifier())]
parameters = {'weights':[[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]}
scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='hard'),parameters,scoring=scorer)

grid_fit = grid_obj.fit(X,y)

best_clf_sv = grid_fit.best_estimator_ 

pred = best_clf_sv.predict(test_X)   
pred
y_out = [[test['id'][i],pred[i]] for i in range(len(pred))]

out_df = pd.DataFrame(data=y_out,columns=['id','class'])

out_df.to_csv(r'out_19.csv',index=False)