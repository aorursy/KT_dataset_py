import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/eval-lab-2-f464/train.csv")
df['chem_0'].fillna(value=df['chem_0'].mean(),inplace=True)

df['chem_1'].fillna(value=df['chem_1'].mean(),inplace=True)

df['chem_2'].fillna(value=df['chem_2'].mean(),inplace=True)

df['chem_3'].fillna(value=df['chem_3'].mean(),inplace=True)

df['chem_4'].fillna(value=df['chem_4'].mean(),inplace=True)

df['chem_5'].fillna(value=df['chem_5'].mean(),inplace=True)

df['chem_6'].fillna(value=df['chem_6'].mean(),inplace=True)

df['chem_7'].fillna(value=df['chem_7'].mean(),inplace=True)

df['attribute'].fillna(value=df['attribute'].mean(),inplace=True)
X_train=df.copy()

X_train.drop(columns=['id','class','chem_7','chem_2','chem_3'],inplace=True)

y_train=df['class']
from sklearn.preprocessing import PowerTransformer



scaler = PowerTransformer()  #Instantiate the scaler 

X_train = scaler.fit_transform(X_train)  #Fit and transform the features using scaler
from sklearn.ensemble import ExtraTreesClassifier

xt = ExtraTreesClassifier(n_estimators=2000)

xt.fit(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=2000)

rf.fit(X_train,y_train)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[('xt', xt), ('rf', rf), ('xgb', model)], voting='soft')

eclf1 = eclf1.fit(X_train, y_train)
df2=pd.read_csv("../input/eval-lab-2-f464/test.csv")
df2['chem_0'].fillna(value=df2['chem_0'].mean(),inplace=True)

df2['chem_1'].fillna(value=df2['chem_1'].mean(),inplace=True)

df2['chem_2'].fillna(value=df2['chem_2'].mean(),inplace=True)

df2['chem_3'].fillna(value=df2['chem_3'].mean(),inplace=True)

df2['chem_4'].fillna(value=df2['chem_4'].mean(),inplace=True)

df2['chem_5'].fillna(value=df2['chem_5'].mean(),inplace=True)

df2['chem_6'].fillna(value=df2['chem_6'].mean(),inplace=True)

df2['chem_7'].fillna(value=df2['chem_7'].mean(),inplace=True)

df2['attribute'].fillna(value=df2['attribute'].mean(),inplace=True)
X_test=df2.copy()

X_test.drop(columns=['id','chem_7','chem_2','chem_3'],inplace=True)
X_test= scaler.transform(X_test)
y_test=eclf1.predict(X_test)

# y_test=np.rint(y_test)

# unique, counts = np.unique(y_test, return_counts=True)

# np.asarray((unique, counts)).T
submission = pd.DataFrame({'id':df2['id'],'class':y_test})

submission['class']=submission['class'].astype("int")

submission['class'].dtype
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)