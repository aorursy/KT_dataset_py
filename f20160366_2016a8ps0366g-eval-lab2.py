import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
df.head()
df.info()
missing_count = df.isnull().sum()
missing_count[missing_count > 0]
missing_count
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)
df_dtype_nunique.columns = ["dtype","unique"]
df_dtype_nunique
df.isnull().any().any()
df.columns
plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(),cmap='Blues',annot=True)
#num_features = ['chem_0', 'chem_1', 'chem_4', 'chem_5',
 #      'chem_6']
num_features = ['chem_0', 'chem_1','chem_2','chem_3', 'chem_4', 
    'chem_6','chem_7','attribute']
    
#categorical_features = ['type']
X= df[num_features]
y = df["class"]
X.head()
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
scaler = RobustScaler()
scaler1 = MinMaxScaler()

# num_features = ['feature5']
# x_train = df.drop(['feature5'],axis=1)[0:4547]
# x_test = df.drop(['feature5'],axis=1)[4547:]

# num_features_scaling= [x for x in numerical_features if x not in num_features]

# x_train_scaled = x_train
# x_test_scaled = x_test

# x_train_scaled[num_features_scaling] = scaler.fit_transform(x_train[num_features_scaling])
# x_test_scaled[num_features_scaling] = scaler.transform(x_test[num_features_scaling])


#scaler = StandardScaler()
#X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
#X_val[numerical_features] = scaler.transform(X_val[numerical_features])  

# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc
#We donot want to train model on our dataset so ,we just want to do the preprocessing while we want to do the preprocessing as well as train our model on our training set . 
#X_train[numerical_features].head()

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(X,y,test_size = 0.9,random_state=42)

x_train_scaled = x_train
x_val_scaled = x_val

x_train_scaled[num_features] = scaler.fit_transform(x_train[num_features])
x_val_scaled[num_features] = scaler.transform(x_val[num_features])

x_scaled=scaler.fit_transform(X[num_features])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


#clf2 = RandomForestClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=30,bootstrap=False).fit(x_train,y_train) # also best
clf3 = ExtraTreesClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=110,bootstrap=False,random_state=None).fit(x_train[num_features],y_train) # also best
clf4 = RandomForestClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=15,bootstrap=False).fit(x_train_scaled,y_train) 
clf5= ExtraTreesClassifier(n_estimators=10, bootstrap=True, oob_score=True,  class_weight='balanced',random_state=None).fit(x_train,y_train)
clf6= ExtraTreesClassifier(n_estimators=10, bootstrap=True, oob_score=True,  class_weight='balanced',random_state=None).fit(x_train_scaled,y_train)

from sklearn.metrics import accuracy_score  #Find out what is accuracy_score
y_pred_3 = clf3.predict(x_val[num_features])
y_pred_4 = clf4.predict(x_val_scaled)
y_pred_5= clf5.predict(x_val)
y_pred_6= clf6.predict(x_val_scaled)
#y_pred=rf.predict(X_val)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_val, y_pred_3))
print(accuracy_score(y_val, y_pred_4))
print(accuracy_score(y_val, y_pred_5))
print(accuracy_score(y_val, y_pred_6))
#xgboost classifier 
#from xgboost import XGBClassifier
#rf=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bynode=1, colsample_bytree=1, gamma=0.2, learning_rate=0.01,
#       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#       n_estimators=100, n_jobs=1, nthread=None,
#       objective='multi:softprob', random_state=1, reg_alpha=0.01,
#       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#       subsample=1, verbosity=1)
#rf.fit(X_train,y_train)

#from sklearn.metrics import confusion_matrix
#y_pred=rf.predict(X_val)
#cm=confusion_matrix(y_val,y_pred)

#from sklearn.model_selection import cross_val_score
#accuracies=cross_val_score(estimator=rf,X=X_train,y=y_train,cv=4)
#accuracies.mean()
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV


#param_distributions=[{'bootstrap': [False,True],
# 'max_depth': [100,150],
# 'max_features': ['auto', 'sqrt'],
# 'min_samples_leaf': [ 4,6],
# 'min_samples_split': [ 5, 10],
# 'n_estimators': [ 3000, 4000]}]
#rf_random=GridSearchCV(estimator=rf,param_grid=param_distributions,cv=3,n_jobs=-1)
#rf_random.fit(X_train,y_train)
#gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
# min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.7,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
# param_grid = param_test7,n_jobs=4,iid=False, cv=5)
#gsearch1.fit(X_train,y_train)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
param_distributions=[{'bootstrap': [False,True],
 'max_depth': [30,50],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1,2],
 'min_samples_split': [4,5],
 'n_estimators': [450,550]}]
rf_random=GridSearchCV(estimator=clf,param_grid=param_distributions,cv=3,n_jobs=-1)
rf_random.fit(x_train_scaled,y_train)
rf_random.best_params_
df1=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
df1.fillna(value=df.mean(),inplace=True)
#X_test_numerical_features = ['chem_0', 'chem_1', 'chem_4', 'chem_5',
     #  'chem_6']
#X_test_categorical_features = ['type']
X_test = df1[num_features]


x_scaled=scaler.fit_transform(X[num_features])
x_test_scaled=scaler.transform(df1[num_features])

#type_code = {'old':0,'new':1}
#X_test['type'] = X_test['type'].map(type_code)

#X_test = pd.get_dummies(data=X_test,columns=['type'])
#X_test.head()

#scaler = StandardScaler()
#X_test[X_test_numerical_features] = scaler.fit_transform(X_test[X_test_numerical_features])

#X_test[X_test_numerical_features].head()
from sklearn.ensemble import ExtraTreesClassifier
#two best models
clf1=ExtraTreesClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=40,bootstrap=False).fit(X[num_features],y) # also best
clf2=ExtraTreesClassifier(n_estimators=500,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=30,bootstrap=True).fit(X[num_features],y)

pred1=clf1.predict(X_test[num_features])
pred2=clf2.predict(X_test[num_features])
pred1
pred2
#For first model 
df1['class']=np.array(pred1)
#For second model
#df1['class']=np.array(pred2)
df1.head()
out=df1[['id','class']]
#out=out.round({'class': 0})

out.head()
out.to_csv('submit_eval_lab_two26.csv',index=False)