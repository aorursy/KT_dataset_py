import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
path = "/kaggle/input/eval-lab-1-f464-v2/train.csv"
df = pd.read_csv(path)
df.head()
y = df['rating']
y
X = df.drop('rating',axis=1)
X = X.drop('id',axis=1)
X.head()
X = pd.get_dummies(X, columns=['type'])
X.head()
X = X.fillna( X.mean() )
X.isnull().any().any()
X['feature3'] = np.log( X['feature3'] )
X['feature5'] = np.log( X['feature5'] )
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_train.shape
#y_train.shape
X.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=2000)

#clf = RandomForestRegressor(n_estimators = 3000, random_state = 42)

clf.fit(X, y);

#pred = clf.predict(X_test)
#pred = np.array( [round(p) for p in pred] )
#rms = sqrt(mean_squared_error(pred , y_test))
#print(rms)
df_test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
df_test.isnull().any()
df_test = df_test.fillna( df_test.mean() )
df_test.isnull().any()
X_ans = df_test.drop('id',axis=1)

X_ans = pd.get_dummies(X_ans, columns=['type'])
X_ans.head()
X_ans['feature3'] = np.log( X_ans['feature3'] )
X_ans['feature5'] = np.log( X_ans['feature5'] )
pred = clf.predict(X_ans)
pred = np.array( [ int(round(p)) for p in pred] )
pred
res = pd.DataFrame(pred, columns=['rating'])
res = pd.concat([df_test['id'],res],axis=1)
res.head()
res.to_csv('submission.csv',index=False)
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=2000)

#clf = (n_estimators=700, max_depth=None, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
rms = sqrt(mean_squared_error(pred , y_test))
print( rms )
'''
'''
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(n_estimators=700, max_depth=None, random_state=0)
parameters = {'n_estimators' : [20,200,500,5000], 
              'max_depth'    : [12,25,200,1000,4000]   }
gs = GridSearchCV(clf, parameters, cv=3)
gs.fit(X_train, y_train)
'''
'''
pred = gs.predict(X_test)
rms = sqrt(mean_squared_error(pred , y_test))
print( rms )
'''
#gs.best_params_