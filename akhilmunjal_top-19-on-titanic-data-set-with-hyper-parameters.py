import pandas as pd

import numpy as np

import seaborn as sns
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
print('Train set : ', df_train.shape,' Test set : ',df_test.shape)
df_train.head()
df_tr = df_train.drop(['PassengerId'],axis=1)

df_tst = df_test.drop(['PassengerId'],axis=1)
numerical_col = df_tr.describe().columns

numerical_col
def missing_data(df):

    total_miss = pd.isnull(df).sum().sort_values(ascending=False)

    percent_miss = ((pd.isnull(df).sum() / pd.isnull(df).count())*100).sort_values(ascending=False)

    missing_values =  pd.concat([total_miss,percent_miss],axis = 1, keys = ['Total', 'Percent'])

    return missing_values
print('Train set :')

missing_data(df_tr)
print('Test set :')

missing_data(df_tst)
df_tr.drop(['Cabin'],axis=1,inplace=True)

df_tst.drop(['Cabin'],axis=1,inplace=True)
df_tr.describe()
df_tr['Embarked'].fillna(df_tr['Embarked'].mode().values[0],inplace=True)

df_tst['Fare'].fillna(df_tst['Fare'].mean(),inplace=True)
def title(df):

    df['Title'] = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]

    columns = df['Title'].unique()

    df['Title'].replace(['Ms','Mlle','Mme'],'Miss',inplace=True)

    df['Title'].replace(['Lady','the Countess','Dona'],'Mrs',inplace=True)

    df['Title'].replace(['Don','Jonkheer','Rev','Sir'],'Unknown',inplace=True)

    df['Title'].replace(['Capt','Col','Dr','Major'],'Mr',inplace=True)

    tile_cols = pd.crosstab(df['Title'],df['Sex'])

    return df.head()
# Ms, Miss, Mlle, Mme ------- Miss

# mrs, lady, countess ------- Mrs

# mr, capt, col, major, dr -- Mr

# don, jonkheer, rev, sir --- Unknown
title(df_tr)
title(df_tst)
def drop_col(df):

    df.drop(['Name','Ticket'],axis=1,inplace=True)
drop_col(df_tr)
drop_col(df_tst)
df_tr.head()
def age_imputation(df):

    for i in ['Master','Mrs','Mr','Unknown','Miss']:

        df[(df['Title']== i) & df['Age'].isna()] = df[(df['Title']== i) 

                                            & df['Age'].isna()].fillna((df['Age'][(df['Title']== i)].mean()))

    
#df_tr[(df_tr['Title']=='Master') & df_tr['Age'].isna()] = df_tr[(df_tr['Title']=='Master') & df_tr['Age'].isna()].fillna((df_tr['Age'][(df_tr['Title']=='Master')].mean()))

#df_tr[(df_tr['Title']=='Mrs') & df_tr['Age'].isna()] = df_tr[(df_tr['Title']=='Mrs') & df_tr['Age'].isna()].fillna((df_tr['Age'][(df_tr['Title']=='Mrs')].mean()))

#df_tr[(df_tr['Title']=='Mr') & df_tr['Age'].isna()] = df_tr[(df_tr['Title']=='Mr') & df_tr['Age'].isna()].fillna((df_tr['Age'][(df_tr['Title']=='Mr')].mean()))

#df_tr[(df_tr['Title']=='Miss') & df_tr['Age'].isna()] = df_tr[(df_tr['Title']=='Miss') & df_tr['Age'].isna()].fillna((df_tr['Age'][(df_tr['Title']=='Miss')].mean()))

#df_tr[(df_tr['Title']=='Unknown') & df_tr['Age'].isna()] = df_tr[(df_tr['Title']=='Unknown') & df_tr['Age'].isna()].fillna((df_tr['Age'][(df_tr['Title']=='Unknown')].mean()))
age_imputation(df_tr)

age_imputation(df_tst)
def age_null(df):

    print( df.isnull().sum().max() )
age_null(df_tr)
age_null(df_tst)
sex_mapping = {"male": 0, "female": 1}

df_tr['Sex'] = df_tr['Sex'].map(sex_mapping)

df_tst['Sex'] = df_tst['Sex'].map(sex_mapping)
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_tr['Embarked'] = df_tr['Embarked'].map(embarked_mapping)

df_tst['Embarked'] = df_tst['Embarked'].map(embarked_mapping)
title_mapping = {"Mr":1,"Mrs":2,"Master":3,"Miss":4,"Unknown":5}

df_tr['Title'] = df_tr['Title'].map(title_mapping)

df_tst['Title'] = df_tst['Title'].map(title_mapping)
df_tr.head()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
df_trn_scaled = df_tr.copy()
df_trn_scaled[['Age','Fare']] = scale.fit_transform(df_trn_scaled[['Age','Fare']])
df_tst[['Age','Fare']] = scale.fit_transform(df_tst[['Age','Fare']])
from sklearn.model_selection import train_test_split



X = df_trn_scaled.drop(['Survived'],axis=1)

y = df_trn_scaled['Survived']



X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.22, random_state = 0)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier(criterion='entropy',random_state=0)

id3.fit(X_train, y_train)

y_pred = id3.predict(X_val)

print('ID3 : ',cross_val_score(id3,X_train,y_train,cv=5), ' Mean : ',cross_val_score(id3,X_train,y_train,cv=5).mean())
cart = DecisionTreeClassifier(criterion='gini',random_state=0)

cart.fit(X_train, y_train)

y_pred = cart.predict(X_val)

print('CART : ',cross_val_score(cart,X_train,y_train,cv=5), ' Mean : ',cross_val_score(cart,X_train,y_train,cv=5).mean())
max_depth=[3,4,5,6,7,8]

min_samples_split=[int(x) for x in np.linspace(start=5,stop=15,num=5)]

max_features=['auto','sqrt','log2']



params = {

    'max_depth' : max_depth,

    'min_samples_split' : min_samples_split,

    'max_features' : max_features

}
clf = RandomizedSearchCV(estimator=id3,param_distributions=params,n_iter=10,n_jobs=-1,random_state=0,cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_
id3_hyp = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='log2',

                       min_samples_split=12, random_state=0).fit(X_train,y_train)

y_pred = id3_hyp.predict(X_val)

print('ID3 Hyp: ',cross_val_score(id3_hyp,X_train,y_train,cv=5), ' Mean : ',cross_val_score(id3_hyp,X_train,y_train,cv=5).mean())
clf = RandomizedSearchCV(estimator=cart,param_distributions=params,n_iter=10,n_jobs=-1,random_state=0,cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_
cart_hyp = DecisionTreeClassifier(criterion='gini',max_depth=6, max_features='sqrt', min_samples_split=12,

                       random_state=0).fit(X_train,y_train)

y_pred = cart_hyp.predict(X_val)

print('CART Hyp: ',cross_val_score(cart_hyp,X_train,y_train,cv=5), ' Mean : ',cross_val_score(cart_hyp,X_train,y_train,cv=5).mean())
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0).fit(X_train,y_train)

y_pred = rfc.predict(X_val)

print('Random Forest : ',cross_val_score(rfc,X_train,y_train,cv=5), ' Mean : ',cross_val_score(rfc,X_train,y_train,cv=5).mean())
max_depth=[3,4,5,6,7,8]

min_samples_split=[int(x) for x in np.linspace(start=5,stop=15,num=5)]

max_features=['auto','sqrt','log2']

oob_score = ['True','False']

n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]





params = {

    'max_depth' : max_depth,

    'min_samples_split' : min_samples_split,

    'max_features' : max_features,

    'oob_score' : oob_score,

    'n_estimators' : n_estimators

}
clf = RandomizedSearchCV(estimator=rfc,param_distributions=params,n_iter=10,n_jobs=-1,random_state=0,cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_
rfc_hyp = RandomForestClassifier(max_depth=4, max_features='log2', min_samples_split=15,

                       n_estimators=500, oob_score='False', random_state=0).fit(X_train,y_train)

y_pred = rfc_hyp.predict(X_val)

print('RF Hyp : ',cross_val_score(rfc_hyp,X_train,y_train,cv=5), ' Mean : ',cross_val_score(rfc_hyp,X_train,y_train,cv=5).mean())
import xgboost as xgb
xgb_cl = xgb.XGBClassifier(random_state=0).fit(X_train,y_train)

y_pred = xgb_cl.predict(X_val)

print('XGBoost : ',cross_val_score(xgb_cl,X_train,y_train,cv=5), ' Mean : ',cross_val_score(xgb_cl,X_train,y_train,cv=5).mean())
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]

max_depth=[3,4,5,6,7,8]

learning_rate = [0.05,0.1,0.15,0.2,0.25,0.3]

colsample_bytree = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

gamma = [0.01,0.03,0.05,0.07,0.1]

reg_lambda = [0.01,0.03,0.05,0.07,0.1]



params = {

    'max_depth' : max_depth,

    'n_estimators' : n_estimators,

    'learning_rate' : learning_rate,

    'colsample_bytree' : colsample_bytree,

    'gamma' : gamma,

    'reg_lambda' : reg_lambda

}
clf = RandomizedSearchCV(estimator=xgb_cl,param_distributions=params,n_iter=10,n_jobs=-1,random_state=0,cv=5)

clf.fit(X_train,y_train)

clf.best_params_
xgb_hyp = xgb.XGBClassifier(n_estimators=200,reg_lambda=0.05,max_depth=3,learning_rate=0.1,gamma=0.01,colsample_bytree=0.5)

xgb_hyp.fit(X_train,y_train)

y_pred = xgb_hyp.predict(X_val)

print('XGBoost Hyp: ',cross_val_score(xgb_hyp,X_train,y_train,cv=5), ' Mean : ',cross_val_score(xgb_hyp,X_train,y_train,cv=5).mean())
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)

y_pred = gbm.predict(X_val)

print('GBM : ',cross_val_score(gbm,X_train,y_train,cv=5), ' Mean : ',cross_val_score(gbm,X_train,y_train,cv=5).mean())
from sklearn.svm import SVC
svc = SVC(random_state=0).fit(X_train, y_train)

y_pred = svc.predict(X_val)

print('SVC : ',cross_val_score(svc,X_train,y_train,cv=5), ' Mean : ',cross_val_score(svc,X_train,y_train,cv=5).mean())
C = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel = ['linear','rbf','poly']

degree = [3,4,5,6]

gamma = ['scale','auto']



params = {

    'C' : C,

    'kernel' : kernel,

    'degree' : degree,

    'gamma' : gamma

}
clf = RandomizedSearchCV(estimator=svc,param_distributions=params,n_iter=10,n_jobs=-1,random_state=0,cv=5)

clf.fit(X_train,y_train)

clf.best_params_
svc_hyp = SVC(kernel='rbf',gamma='auto',C=1,random_state=0).fit(X_train, y_train)

y_pred = svc_hyp.predict(X_val)

print('SVC Hyp : ',cross_val_score(svc_hyp,X_train,y_train,cv=5), ' Mean : ',cross_val_score(svc_hyp,X_train,y_train,cv=5).mean())
ids = df_test['PassengerId']



# ID3 predictions

predictions_id3 = id3.predict(df_tst)

output_id3 = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_id3 })

output_id3.to_csv('submission_id3.csv', index=False)



# ID3 HYP predictions

predictions_id3_hyp = id3_hyp.predict(df_tst)

output_id3_hyp = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_id3_hyp })

output_id3_hyp.to_csv('submission_id3_hyp.csv', index=False)



# CART predictions

predictions_cart = cart.predict(df_tst)

output_cart = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_cart })

output_cart.to_csv('submission_cart.csv', index=False)



# CART HYP predictions

predictions_cart_hyp = cart_hyp.predict(df_tst)

output_cart_hyp = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_cart_hyp })

output_cart_hyp.to_csv('submission_cart_hyp.csv', index=False)



# Random Forest predictions

predictions_rfc = rfc.predict(df_tst)

output_rfc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_rfc })

output_rfc.to_csv('submission_rfc.csv', index=False)



# Random Forest HYP predictions

predictions_rfc_hyp = rfc_hyp.predict(df_tst)

output_rfc_hyp = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_rfc_hyp })

output_rfc_hyp.to_csv('submission_rfc_hyp.csv', index=False)



# XGBoost predictions

predictions_xgb = xgb_cl.predict(df_tst)

output_xgb = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_xgb })

output_xgb.to_csv('submission_xgb.csv', index=False)



# XGBoost HYP predictions

predictions_xgb_hyp = xgb_hyp.predict(df_tst)

output_xgb_hyp = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_xgb_hyp })

output_xgb_hyp.to_csv('submission_xgb_hyp.csv', index=False)



# GBM predictions

predictions_gbm = gbm.predict(df_tst)

output_gbm = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_gbm })

output_gbm.to_csv('submission_gbm.csv', index=False)



# SVC predictions

predictions_svc = svc.predict(df_tst)

output_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_svc })

output_svc.to_csv('submission_svc.csv', index=False)



# SVC HYP predictions

predictions_svc_hyp = svc_hyp.predict(df_tst)

output_svc_hyp = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_svc_hyp })

output_svc_hyp.to_csv('submission_svc_hyp.csv', index=False)


