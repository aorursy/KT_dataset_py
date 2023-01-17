import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head(10)
test.head()
train['knockedOutSoldiers'] = train['knockedOutSoldiers'].fillna(0)
train['respectEarned'] = train['respectEarned'].fillna(np.mean(train['respectEarned']))
train['respectEarned'] = round(train['respectEarned'])
train_new.head()
train.shape
test.shape
train.dtypes
train.shape
train_new = train
train_new.shape
count = 0

for i in range(3,24):
        count += 1
        str_temp = "col"
        str_temp = str_temp + str(count)
        train_new[str_temp] = train.iloc[:,i]*train.iloc[:,7]
train_new.shape
train.shape
test.head()
test_new = test
test_new.shape
count = 0

for i in range(3,24):
        count += 1
        str_temp = "col"
        str_temp = str_temp + str(count)
        test_new[str_temp] = test.iloc[:,i]*test.iloc[:,7]
test_new.head()
test_new.shape
cols = ['soldierId', 'bestSoldierPerc']
cols_test = ['soldierId']
random_rows=list(np.random.random_integers(0, train.shape[0], 20000))
train_new = train.iloc[random_rows,:]
train_new.shape
X = train_new.drop(cols,axis=1)
X.shape
X_t = test.drop(cols_test,axis=1)
X_t.shape
X.head()
y = train_new['bestSoldierPerc']
y.shape
#X_t=test.iloc[:,1:256]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
clsf = xgb.XGBRegressor(n_estimators=100, learning_rate=0.09, gamma = 0, subsample=1,
                         colsample_bytree=1, max_depth=7, min_child_weight = 1).fit(X_train, y_train, sample_weight=None)
predictions = clsf.predict(X_test)
np.mean(abs(predictions-y_test))



feature_importance=clsf.feature_importances_
len(feature_importance)
#list_nouse = []
#cols_of_value = []
for i in range(X.shape[1]):
   # if i < 25 :
    #    continue
    #if feature_importance[i] < 0.02:
     #   list_nouse.append(X.columns[i])
    #else:
     #   cols_of_value.append(X.columns[i])
    print(X.columns[i],':\t',feature_importance[i])
X_t.shape
test_new.shape
predtest = clsf.predict(X_t)
sol=pd.DataFrame()
sol['soldierId']=test.soldierId
pred= pd.DataFrame(predtest)
pred.head()
sol['bestSoldierPerc']=pred.iloc[:,0]
sol.head()
sol.shape
test.shape
sol.to_csv('submission_file.csv',index=False)





