import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
train=pd.read_csv('../input/train.csv')
trainheatmap=pd.concat([train['Elevation'],train['Aspect'],train['Slope'],
                       train['Horizontal_Distance_To_Hydrology'],
                       train['Vertical_Distance_To_Hydrology'],
                       train['Horizontal_Distance_To_Roadways'],
                       train['Hillshade_9am'],train['Hillshade_Noon'],
                       train['Hillshade_3pm'],train['Horizontal_Distance_To_Fire_Points']],axis=1)
sns.heatmap(np.abs(trainheatmap.corr()))
plt.show()
X=pd.DataFrame()
traincolumnlist=[]
Y=train['Cover_Type']
column=train.columns
for i in range(1,len(column)-1):
    X=pd.concat([X,train[column[i]]],axis=1)
    traincolumnlist.append(column[i])
X.columns=traincolumnlist
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
gbm=xgb.XGBClassifier(max_depth=5,
                     learning_rate=0.05,
                     n_estimators=50,
                     objective='multi:softprob',
                     booster='gbtree',
                     silent=1).fit(X_train,Y_train,verbose=True)
prediction1 = gbm.predict(X_train)
gbm1 = accuracy_score(Y_train, prediction1)
print('train:'+str(gbm1))
prediction2 = gbm.predict(X_test)
gbm2 = accuracy_score(Y_test, prediction2)
print('test:'+str(gbm2))
test=pd.read_csv('../input/test.csv')
testX=pd.DataFrame()
for i in range(1,len(column)-1):
    testX=pd.concat([testX,test[column[i]]],axis=1)
testY=gbm.predict(testX)
testresult=pd.concat([test['Id'],pd.Series(testY)],axis=1)
testresult.columns=['Id','Cover_Type']
testresult.to_csv('submission.csv',index=False)