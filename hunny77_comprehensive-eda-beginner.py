import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, classification_report
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', None)
train = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
test = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
sample = pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv")
train.info()
train.head()
train.isna().sum()
test.isna().sum()
for col in train.columns:
    print(f"{col} : {train[col].nunique()}")
    print(train[col].unique())
#separating continuous and categorical variables
cat_var = ["Gender","Driving_License","Previously_Insured","Vehicle_Age","Vehicle_Damage"]
con_var = list(set(train.columns).difference(cat_var+["Response"]))
train.Response.value_counts(normalize=True)
sns.countplot(train.Response)
plt.title("Class count")
plt.show()

train.head(3)
# axis is as follows

sns.pairplot(train, hue='Response', diag_kind='hist')
plt.show()
def map_val(data):
    data["Gender"] = data["Gender"].replace({"Male":1, "Female":0})
    data["Vehicle_Age"] = data["Vehicle_Age"].replace({'> 2 Years':2, '1-2 Year':1, '< 1 Year':0 })
    data["Vehicle_Damage"] = data["Vehicle_Damage"].replace({"Yes":1, "No":0})
    return data

train = map_val(train)
test = map_val(test)
comb = pd.concat([train,test])
comb.shape , train.shape , test.shape

print('The distribution of gender:',comb['Gender'].value_counts())
comb.head()
comb.info()
list1 = ['Gender', 'Age', 'Region_Code', 'Previously_Insured',
       'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']
# list1 = set(comb.columns) - set('Driving_License')
list1

fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(20,20))

for i, column in enumerate(list1):
    print(column)
    sns.distplot(comb[column],ax=axes[i//3,i%3])
train.head(3)
cat_var
fig, ax = plt.subplots(2,3 , figsize=(15,15))
ax = ax.flatten()
for i,col in enumerate(cat_var):
    sns.pointplot(x = col, y = 'Response',hue = 'Vehicle_Age',data=train, ax = ax[i])
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(2,3 , figsize=(10,10))
ax = ax.flatten()
for i,col in enumerate(cat_var):
    sns.pointplot(col, 'Response', data=train, ax = ax[i])
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Response',hue='Vehicle_Age', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='point', height=3, aspect=2)
plt.show()
fig, ax = plt.subplots(2,3 , figsize=(16,6))
ax = ax.flatten()
i = 0
for col in con_var:
    sns.boxplot( 'Response', col, data=train, ax = ax[i])
    i+=1
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Vintage',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Age',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Annual_Premium',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
plt.figure(figsize=(30,5))
sns.heatmap(pd.crosstab([train['Previously_Insured'], train['Vehicle_Damage']], train['Region_Code'],
                        values=train['Response'], aggfunc='mean', normalize='columns'), annot=True, cmap='inferno')
plt.show()
crosstab_df=pd.crosstab([train['Previously_Insured'], train['Vehicle_Damage']], train['Region_Code'],values=train['Response'], aggfunc='mean', normalize='columns')
crosstab_df
cat_var
train.head(1)
sns.relplot(x="Age", y="Annual_Premium", hue="Response", data=train)
sns.relplot(x="Vintage", y="Annual_Premium", hue="Response", data=train)
sns.relplot(x="Policy_Sales_Channel", y="Annual_Premium", hue="Response", data=train)
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
plt.title("Correlation Heatmap")
plt.show()
train.skew()
train['log_premium'] = np.log(train.Annual_Premium)
train['log_age'] = np.log(train.Age)
test['log_premium'] = np.log(test.Annual_Premium)
test['log_age'] = np.log(test.Age)
train.groupby(['Previously_Insured','Gender'])['log_premium'].plot(kind='kde')
plt.show()
train.groupby(['Previously_Insured','Gender'])['log_age'].plot(kind='kde')
plt.show()
import eli5
from eli5.sklearn import PermutationImportance
def feature_engineering(data, col):
    mean_age_insured = data.groupby(['Previously_Insured','Vehicle_Damage'])[col].mean().reset_index()
    mean_age_insured.columns = ['Previously_Insured','Vehicle_Damage','mean_'+col+'_insured']
    mean_age_gender = data.groupby(['Previously_Insured','Gender'])[col].mean().reset_index()
    mean_age_gender.columns = ['Previously_Insured','Gender','mean_'+col+'_gender']
    mean_age_vehicle = data.groupby(['Previously_Insured','Vehicle_Age'])[col].mean().reset_index()
    mean_age_vehicle.columns = ['Previously_Insured','Vehicle_Age','mean_'+col+'_vehicle']
    data = data.merge(mean_age_insured, on=['Previously_Insured','Vehicle_Damage'], how='left')
    data = data.merge(mean_age_gender, on=['Previously_Insured','Gender'], how='left')
    data = data.merge(mean_age_vehicle, on=['Previously_Insured','Vehicle_Age'], how='left')
    data[col+'_mean_insured'] = data['log_age']/data['mean_'+col+'_insured']
    data[col+'_mean_gender'] = data['log_age']/data['mean_'+col+'_gender']
    data[col+'_mean_vehicle'] = data['log_age']/data['mean_'+col+'_vehicle']
    data.drop(['mean_'+col+'_insured','mean_'+col+'_gender','mean_'+col+'_vehicle'], axis=1, inplace=True)
    return data

train = feature_engineering(train, 'log_age')
test = feature_engineering(test, 'log_age')

train = feature_engineering(train, 'log_premium')
test = feature_engineering(test, 'log_premium')

train = feature_engineering(train, 'Vintage')
test = feature_engineering(test, 'Vintage')
train
test
X = train.drop(["Response"], axis=1)
Y = train["Response"]
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.2,random_state=294,stratify = Y)
lg=LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=5,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)

lg.fit(X_train,y_train)
print(roc_auc_score(y_val,lg.predict_proba(X_val)[:,1]))
#Check for Permutation Importance of Features
perm = PermutationImportance(lg,random_state=294).fit(X_val, y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())

def drop_permute_feat(data,permuter):
    mask = permuter.feature_importances_ > 0 
    features = data.columns[mask]
    return features
features = drop_permute_feat(X_train,perm)
features
X_train_permute = X_train[features]
X_val_permute = X_val[features]
lg=LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=5,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)

lg.fit(X_train_permute,y_train)
print(roc_auc_score(y_val,lg.predict_proba(X_val_permute)[:,1]))
## Full fit
lg=LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)
lg.fit(X,Y)
submission_df=pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv')
submission_df['Response']=np.array(lg.predict_proba(test)[:,1])
submission_df.to_csv('baseline_test.csv',index=False)
submission_df.head(5)
def drop(data,list2):
    data_new = data.drop(list2, axis=1,inplace = False)
    return data_new





X.info()
test.info()
X_select = X.copy()
test_select = test.copy()
from sklearn.model_selection import StratifiedKFold
model_xgb = XGBClassifier(n_jobs=4, random_state=1, scale_pos_weight=7, objective='binary:logistic')
model_lgbm = LGBMClassifier(n_jobs=4, random_state=1, is_unbalance=True, objective='binary')
model_cat = CatBoostClassifier(random_state=1, verbose=0, scale_pos_weight=7, custom_metric=['AUC'])
def submission(preds, model):
    sample["Response"] = preds
    sample.to_csv("model_"+model+".csv", index=False)
model_lgbm = LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)
def cv_generator(model,n_splits_user,X_select,Y):
    cv = StratifiedKFold(n_splits=n_splits_user, random_state=1, shuffle=True)
    predictions= []
    train_roc_score = 0
    test_roc_score = 0

    for train_index, test_index in cv.split(X_select, Y):
        xtrain, xtest = X_select.iloc[train_index], X_select.iloc[test_index]
        ytrain, ytest = Y[train_index], Y[test_index]

        model.fit(xtrain, ytrain)
        trainpred = model.predict_proba(xtrain)[:,1]
        testpred = model.predict_proba(xtest)[:,1]
        train_roc_score += roc_auc_score(ytrain, trainpred)
        test_roc_score += roc_auc_score(ytest, testpred)
        print("Train ROC AUC : %.4f Test ROC AUC : %.4f"%(roc_auc_score(ytrain, trainpred),roc_auc_score(ytest, testpred)))

        prediction = model.predict_proba(test_select)[:,1]
        predictions.append(prediction)
    
    print("The mean train score is :",train_roc_score/5)
    print("The mean test score is :",test_roc_score/5)
    
    return prediction

predictions_lgbm = cv_generator(model = model_lgbm,n_splits_user = 5,X_select = X,Y = Y)
submission(np.mean(predictions_lgbm, axis=0), 'lgbm_stack')
predictions_xgb = cv_generator(model = model_xgb,n_splits_user = 5,X_select = X,Y = Y)
submission(np.mean(predictions_xgb, axis=0), 'xgb_stack')
predictions_cat = cv_generator(model = model_cat,n_splits_user = 5,X_select = X,Y = Y)
submission(np.mean(predictions_cat, axis=0), 'cat_stack')
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.2,random_state=294,stratify = Y)
# categorical column 
cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
X.info()
X.columns
X.Region_Code.dtype == 'float64'

X_copy = X.copy()
test_copy = test.copy()
for col in test.columns:
    if test[col].dtype == 'float64' :
        test_copy[col] = test[col].astype('int')
test_copy.info()          
        
        
for col in X.columns:
    if X[col].dtype == 'float64' :
        X_copy[col] = X[col].astype('int')
        
X_copy.info()        
        
col_1=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X_t, X_tt, y_t, y_tt = train_test_split(X_copy[col_1], Y, test_size=.25, random_state=150303,stratify=Y,shuffle=True)
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
catb = CatBoostClassifier()
catb= catb.fit(X_t, y_t,cat_features=cat_col,eval_set=(X_tt, y_tt),plot=True,early_stopping_rounds=40,verbose=100)
#catb= catb.fit(X_t, y_t,cat_features=cat_col,eval_set=(X_tt, y_tt),plot=True,verbose=100)
y_cat = catb.predict(X_tt)
probs_cat_train = catb.predict_proba(X_t)[:, 1]
probs_cat_test = catb.predict_proba(X_tt)[:, 1]
roc_auc_score(y_t, probs_cat_train)
roc_auc_score(y_tt, probs_cat_test)
cat_pred_new= catb.predict_proba(test_copy[col_1])[:, 1]

submission(cat_pred_new,'cat_boost_predictions_reduced_cols')

feat_importances = pd.Series(catb.feature_importances_, index=X_t.columns)
feat_importances.nlargest(15).plot(kind='barh')
#feat_importances.nsmallest(20).plot(kind='barh')
plt.show()
X_final = X_copy[col_1].copy()
test_final = test_copy[col_1].copy()
X_final.info()
test_final.info()
