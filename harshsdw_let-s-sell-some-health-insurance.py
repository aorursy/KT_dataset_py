import pandas as pd 

df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df.head()
df.info()
X = pd.get_dummies(df)
X.info()
X.Response.value_counts()
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = X[X.Response==0]
df_minority = X[X.Response==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=334399,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Response.value_counts()
y = df_upsampled.Response

df_upsampled.drop(columns=['Response','id'],inplace=True,axis=1)

df_upsampled.rename({'Vehicle_Age_< 1 Year': 'Vehicle_Age_less_than 1 Year' , 'Vehicle_Age_> 2 Years': 'Vehicle_Age_greater_than 2 Years'},axis=1, inplace=True)

df_upsampled.info()
df_upsampled.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_upsampled[df_upsampled.columns] = scaler.fit_transform(df_upsampled[df_upsampled.columns])

df_upsampled.head()
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(df_upsampled , y , test_size = 0.33 , random_state =123)
X_train.shape, X_test.shape
from xgboost import *

xgb = XGBClassifier(max_depth = 6, 
                            n_estimators = 30000, 
                            objective = 'binary:logistic',colsample_bytree = 0.5, gamma = 0.3 , reg_lambda = 1.2, reg_alpha = 1.2, min_child_weight = 1, 
               learning_rate = 0.1,eval_metric = 'auc' ,tree_method='gpu_hist', gpu_id=0).fit(X_train,y_train)


from sklearn.metrics import roc_auc_score 
Cat_preds1 = xgb.predict_proba(X_train)
Cat_class1 = xgb.predict(X_train)
Cat_score1 = roc_auc_score(y_train, Cat_preds1[:,1])
print(Cat_score1)
Cat_preds1 = xgb.predict_proba(X_test)
Cat_class1 = xgb.predict(X_test)
Cat_score1 = roc_auc_score(y_test, Cat_preds1[:,1])
print(Cat_score1)
from catboost import CatBoostClassifier

cat = CatBoostClassifier(iterations = 30000, 
                                random_seed = 69, 
                                 task_type = 'GPU',
                                learning_rate=0.25,
                                depth=8,
                                loss_function='Logloss',
                                bootstrap_type='Poisson',
                                subsample = 0.8,
                                custom_loss = ['AUC'] ).fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
Cat_preds1 = cat.predict_proba(X_train)
Cat_class1 = cat.predict(X_train)
Cat_score1 = roc_auc_score(y_train, Cat_preds1[:,1])
print(Cat_score1)
Cat_preds1 = cat.predict_proba(X_test)
Cat_class1 = cat.predict(X_test)
Cat_score1 = roc_auc_score(y_test, Cat_preds1[:,1])
print(Cat_score1)
feat_importances = pd.Series(cat.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')