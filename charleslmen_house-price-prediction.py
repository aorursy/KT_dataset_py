# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Import train datastet
train = pd.read_csv('../input/train.csv')
train.head()
test=pd.read_csv('../input/test.csv')
test.head()
# Checking for missing values
def find_missing(train,test):
    count_missing_train=train.isnull().sum().values
    count_missing_test=test.isnull().sum().values
    total_train=train.shape[0]
    total_test=test.shape[0]
    ratio_missing_train=count_missing_train/total_train*100
    ratio_missing_test=count_missing_test/total_test*100
    return pd.DataFrame({'Missing_train':count_missing_train,'Missing_Ratio_train':ratio_missing_train,
                        'Missing_test':count_missing_test,'Missing_Ratio_test':ratio_missing_test},
                       index=train.columns)
df_missing=find_missing(train.drop(columns='SalePrice',axis=1),test)
df_missing=df_missing[df_missing['Missing_Ratio_train']>0].sort_values(by='Missing_Ratio_train',ascending=False)
df_missing.head(7)
from sklearn.metrics import r2_score,mean_squared_error
# Print R2 and RMSE scores
def get_score(prediction,labels):
    print('R2:{}'.format(r2_score(prediction,labels)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(prediction,labels))))

# Shows scores of training and validation sets
def train_test(estimator,x_train,x_test,y_train,y_test):
    prediction_train=estimator.predict(x_train)
    print(estimator)
    get_score(prediction_train,y_train)
    prediction_test=estimator.predict(x_test)
    print("TEST")
    get_score(prediction_test,y_test)
# Splitting to features and target
train_target=train.iloc[:,-1].values
features=pd.concat([train.drop(columns='SalePrice'),test],keys=['Train','Test'])
features.head()
# Delete some features which are not correspond to the sales price and also whose features contains lots of missing values
features.drop(columns=['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']
             ,axis=1,inplace=True)
features.head()
numerics=['int16','int32','int64','float16','float32','float64']
df_numerics=features.select_dtypes(include=numerics)
df_numerics.head()
missing_values=[]
for c in df_numerics.columns:
    if df_numerics[c].isnull().sum()>0:
        missing_values.append(c)
missing_values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(df_numerics.loc[:,('LotFrontage','TotalBsmtSF','GarageCars')])
df_numerics.loc[:,('LotFrontage', 'TotalBsmtSF', 'GarageCars')]=imputer.transform(df_numerics.loc[:,('LotFrontage', 'TotalBsmtSF', 'GarageCars')])
df_categorical=features.drop(columns=df_numerics.columns,axis=1)
df_categorical.head()
missing_values=[]
for c in df_categorical.columns:
    if df_categorical[c].isnull().sum()>0:
        missing_values.append(c)
missing_values
# Filling NAs for categorical features
df_categorical['MSZoning']=df_categorical['MSZoning'].fillna(df_categorical['MSZoning'].mode()[0])
df_categorical['Alley']=df_categorical['Alley'].fillna('NOACCESS')
df_categorical['MasVnrType'] = df_categorical['MasVnrType'].fillna(df_categorical['MasVnrType'].mode()[0])
df_categorical['Exterior1st']=df_categorical['Exterior1st'].fillna(df_categorical['Exterior1st'].mode()[0])
df_categorical['Exterior2nd']=df_categorical['Exterior2nd'].fillna(df_categorical['Exterior2nd'].mode()[0])
for c in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_categorical[c] = df_categorical[c].fillna('NoBSMT')
df_categorical['Electrical'] = df_categorical['Electrical'].fillna(df_categorical['Electrical'].mode()[0])
df_categorical['KitchenQual'] = df_categorical['KitchenQual'].fillna(df_categorical['KitchenQual'].mode()[0])
df_categorical['FireplaceQu'] = df_categorical['FireplaceQu'].fillna('NoFP')
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    df_categorical[col] = df_categorical[col].fillna('NoGRG')
df_categorical['SaleType'] = df_categorical['SaleType'].fillna(df_categorical['SaleType'].mode()[0])
features=pd.concat([df_numerics,df_categorical],axis=1)
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
features.head()
sns.distplot(train_target)
# Label features encoding
conditions=set([x for x in features['Condition1']]+[x for x in features['Condition2']])
dummies=pd.DataFrame(data=np.zeros((len(features.index),len(conditions))),
                       index=features.index,columns=conditions)
for i,cond in enumerate(zip(features['Condition1'],features['Condition2'])):
    dummies.ix[i, cond]=1
features=pd.concat([features,dummies.add_prefix('Condition_')],axis=1)
features.drop(columns=['Condition1','Condition2'],axis=1,inplace=True)

exteriors=set([x for x in features['Exterior1st']]+[x for x in features['Exterior2nd']])
dummies=pd.DataFrame(data=np.zeros((len(features.index),len(exteriors))),
                    index=features.index,columns=exteriors)
for i,ext in enumerate(zip(features['Exterior1st'],features['Exterior2nd'])):
    dummies.ix[i,ext]=1
features=pd.concat([features,dummies.add_prefix('Exterior_')],axis=1)
features.drop(columns=['Exterior1st','Exterior2nd'],axis=1,inplace=True)

for c in features.dtypes[features.dtypes=='object'].index:
    for_dummy=features.pop(c)
    features=pd.concat([features,pd.get_dummies(for_dummy,prefix=c)],axis=1)
features.head()
# Using Elastic Net and GBoosing model
features_standard=features.copy()

train_features=features.loc['Train'].drop(columns='Id',axis=1).values
test_features=features.loc['Test'].drop(columns='Id',axis=1).values

train_features_st=features_standard.loc['Train'].drop(columns='Id',axis=1).values
test_features_st=features_standard.loc['Test'].drop(columns='Id',axis=1).values
# Feature scaling for Elastic Net model
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
train_features_st=sc_X.fit_transform(train_features_st)
test_features_st=sc_X.transform(test_features_st)
from sklearn.utils import shuffle
train_features_st,train_features,train_target=shuffle(train_features_st,train_features,train_target,random_state=0)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_features,train_target,test_size=0.2,random_state=0)
x_train_st,x_test_st,y_train_st,y_test_st=train_test_split(train_features_st,train_target,test_size=0.2,random_state=0)
# Elastic Net
from sklearn.linear_model import ElasticNetCV
ENSTest=ElasticNetCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10],
                     l1_ratio=[.01,.1,.5,.9,.99],max_iter=5000).fit(x_train_st,y_train_st)
train_test(ENSTest,x_train_st,x_test_st,y_train_st,y_test_st)
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(ENSTest,train_features_st,train_target,cv=5)
print('Accuracy : %0.2f (+/-%0.2f)' % (accuracy.mean(),accuracy.std()*2))
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
GBTest=GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=3,max_features='sqrt',
                                min_samples_leaf=15,min_samples_split=10,loss='huber').fit(x_train,y_train)
train_test(GBTest,x_train,x_test,y_train,y_test)
accuracy=cross_val_score(GBTest,train_features,train_target,cv=5)
print('Accuracy : %0.2f (+/-%0.2f)' % (accuracy.mean(),accuracy.std()*2))
# Retrain the model on the whole train set
GBModel=GBTest.fit(train_features,train_target)
ENSTModel=ENSTest.fit(train_features_st,train_target)
final=(GBModel.predict(test_features)+ENSTModel.predict(test_features_st))/2
submission=pd.DataFrame({'Id':test['Id'].values})
submission['SalePrice']=final
submission.to_csv('submission.csv',index=False)