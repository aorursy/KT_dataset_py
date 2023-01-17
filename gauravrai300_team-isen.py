import pandas as pd
mydf= pd.read_csv('C:/Users/gaura/Desktop/conoco/equip_failures_training_set.csv', na_values=['na'])
pd.set_option('display.max_columns',None)

mydf.head()
# percentage of Null values in each column

pd.set_option('display.max_rows',None)

(mydf.isnull().sum())/len(mydf)
# Dropping the columns having more than 25000 Null values.

mydf1 = mydf.dropna(axis=1, thresh=25000)
mydf1.shape
y= mydf1['target']

mydf1= mydf1.drop(['id','target'], axis=1)
# imputing Null values with median of each column

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

X= imputer.fit_transform(mydf1)
mydf2= pd.DataFrame(data=X, columns=mydf1.columns)
mydf2.isna().sum()
# mean normalizing the features/ feature scaling

X= (mydf2- mydf2.mean(axis=0))/mydf2.std(axis=0)
X= X.dropna(axis=1)
from xgboost import XGBClassifier
model= XGBClassifier(learning_rate=0.5, n_estimators=500, max_depth=3)
model.fit(X,y)
pred= model.predict(X)
sum(pred)
accuracy= sum(y==pred)/len(X)
accuracy
# checking the accuracy with cross validation.

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

scores = cross_val_score(model,X, y, cv=StratifiedKFold(5),error_score='f1_score' )

scores  
scores.mean()
testdf= pd.read_csv('C:/Users/gaura/Desktop/conoco/equip_failures_test_set.csv', na_values=['na'])
testdf2=  testdf[mydf1.columns]
testX= imputer.transform(testdf2)
testdf2= pd.DataFrame(data=testX, columns=testdf2.columns)
X= (testdf2- mydf2.mean(axis=0))/mydf2.std(axis=0)
X= X.drop( 'sensor54_measure', axis=1)
pred= model.predict(X)
test_result= pd.DataFrame(pred, columns=['target'])

test_result['id']=testdf['id']
test_result.to_csv('C:/Users/gaura/Desktop/conoco/pred.csv')