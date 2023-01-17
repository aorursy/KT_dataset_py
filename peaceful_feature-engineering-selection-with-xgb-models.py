import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import gc

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
id_test = test_df['Id']
print(train_df.shape)
print(test_df.shape)

labels = train_df['Target'].values
train_df.drop('Target',axis=1, inplace=True)
# concatenate both train and test data
alldata = pd.concat([train_df, test_df])
print(alldata.shape)

# encoding categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
alldata['idhogar'] = le.fit_transform(alldata['idhogar'])


# handle missing values
missing_cols = alldata.columns[alldata.isnull().any()].tolist()
print(missing_cols)


# We first examine 'v2a1' which is the monthly rent amount
# Clearly that people who rent a house have to rent, so let's check this probability
alldata[alldata['tipovivi3']==1].shape[0]/alldata.shape[0]
# only 17% people rent a house, so most people do not have to pay
# we therefore set this to be 0
alldata['v2a1'].fillna(0, inplace=True)
# fill missing meaneduc and SQBmeaned
# Clearly, we can use SQBmeaned to infer meaneduc and vice-versa
ids = alldata['SQBmeaned'].notnull() & alldata['meaneduc'].isnull()
alldata.loc[ids, 'meaneduc'] = np.sqrt(alldata[ids]['SQBmeaned'])

ids = alldata['SQBmeaned'].isnull() & alldata['meaneduc'].notnull()
alldata.loc[ids, 'SQBmeaned'] = np.square(alldata[ids]['meaneduc'])
alldata[alldata['meaneduc'].isnull()].shape[0]

# there are still 36 missing values
# we can use the average years of schooling as mean of years of education, intuitively
df=alldata.groupby('idhogar')['escolari'].mean().reset_index().rename(columns={'escolari':'escolari_mean'})
alldata = alldata.merge(df, how='left', on='idhogar')
mean_ids = alldata['meaneduc'].isnull()
sq_mean_ids = alldata['SQBmeaned'].isnull()
alldata.loc[mean_ids, 'meaneduc'] = alldata.loc[mean_ids,'escolari_mean']
alldata.loc[sq_mean_ids,'SQBmeaned'] = np.square(alldata.loc[sq_mean_ids,'meaneduc'])

# let's check possible values of this column
alldata['rez_esc'].value_counts()
# there is only one row with value of 99 which does not seem to be right (indeed, the corresponding age is only 8), let's set it to nan
alldata.loc[alldata['rez_esc']>5, 'rez_esc'] = np.nan
# we could see the the majority has value of 0, but let's see the range of ages for which rez_esc > 0 
#alldata.groupby('rez_esc')['age'].mean().reset_index()
min_age=alldata[alldata['rez_esc']>0]['age'].min()
max_age=alldata[alldata['rez_esc']>0]['age'].max()
print ("min age: {}, max age: {}".format( min_age, max_age))
# easiest way is to set any age outside of the above range to 0
alldata.loc[alldata['rez_esc'].isnull() & ((alldata['age']<8)|(alldata['age']>17)), 'rez_esc'] = 0
missing_left = list(alldata[alldata['rez_esc'].isnull()]['age'])
print(missing_left)

# after filling, we have 5 missing values left
# of course, it would not make much difference which value we choose for this particular instance, so we can choose 0
# but let's examine the probability that a certain age corresponds to a particular value of res_esc
for age in set(missing_left):
    ids = (alldata['age']==age)&(alldata['rez_esc'].notnull())
    df = alldata[ids].groupby('rez_esc')['Id'].count().reset_index()
    df['Id'] = df['Id']/alldata[ids].shape[0]
    #rez_esc_dict.update({age:list(df['Id'])})
    print("age {} : {}".format(age, list(df['Id'])))

# we can see that rez_esc = 0 is the majority in all cases
# so let's set the remaining missing value to 0
alldata['rez_esc'].fillna(0,inplace=True)
# filling missing values for v18q1
# of course, one does not own a tablet should have 0 tablets
alldata.loc[alldata['v18q']==0,'v18q1']=0

# the majority is that one owns and has exactly 1 tablet
alldata.isnull().any().any()

# we remove 'yes' and 'no' from numerical columns using the coressponding square
alldata.loc[alldata['edjefe']=='no','edjefe'] = 0
alldata.loc[alldata['edjefe']=='yes','edjefe'] = 1
alldata['edjefe'] = alldata['edjefe'].astype(int)

alldata.loc[alldata['edjefa']=='no','edjefa'] = 0
alldata.loc[alldata['edjefa']=='yes','edjefa'] = 1
alldata['edjefa'] = alldata['edjefa'].astype(int)

text_ids = (alldata['dependency']=='yes')|(alldata['dependency']=='no')
alldata.loc[text_ids,'dependency'] = np.sqrt(alldata.loc[text_ids,'SQBdependency'])
alldata['dependency'] = alldata['dependency'].astype(float)
# we explore data before performing feature engineering
# here I tried with dependency as an example
# note that dependency rate = (people with age <=19 and >=64)/(people with age within [19,64])
# we create a new column for people within [19,64] called working_people
alldata['working_people'] = alldata['hogar_adul'] - alldata['hogar_mayor']
# we create a columns for people <=19 and >=64
alldata['not_working_people'] = alldata['hogar_nin']+alldata['hogar_mayor']
# it should hold that dependency = not_working_people/working_people, so let's check that
# now, we check that
alldata[['not_working_people','working_people','dependency']]
# oh, they do not agree on many rows
# let's check at which value of dependency we have the difference
alldata[(alldata['not_working_people']%alldata['working_people']!=0)&(np.abs(alldata['not_working_people']-(alldata['working_people']*alldata['dependency']))>0.0001)][['not_working_people','working_people','dependency']]


# interesting! they differ only when dependency = 8 and working_people = 0
# of course, when no people are working then the dependency should be maximal, but in this case the dependency rate should depends on the number of non_working_people
ids = alldata['dependency']>7 # not that comparison with float is dangerous, so we do it in a safer way
alldata.loc[ids, 'dependency'] = alldata.loc[ids,'not_working_people']+8
# add new features
def statsFeature(df, grp_cols, stats_col, stats='sum'):
    prefix = '_'.join(grp_cols+[stats_col])
    new_col = prefix+"_"+stats
    if stats=='min':
        new_df = df.groupby(grp_cols)[stats_col].min().reset_index().rename(columns={stats_col:new_col})
    elif stats=='max':
        new_df = df.groupby(grp_cols)[stats_col].max().reset_index().rename(columns={stats_col:new_col})
    elif stats=='mean':
        new_df = df.groupby(grp_cols)[stats_col].mean().reset_index().rename(columns={stats_col:new_col})
    elif stats=='median':    
        new_df = df.groupby(grp_cols)[stats_col].median().reset_index().rename(columns={stats_col:new_col})
    elif stats=='count':
        new_df = df.groupby(grp_cols)[stats_col].count().reset_index().rename(columns={stats_col:new_col})
    else:
        new_df = df.groupby(grp_cols)[stats_col].sum().reset_index().rename(columns={stats_col:new_col})
    df = df.merge(new_df, how = 'left', on=grp_cols)
    return df

# generate counting features
alldata = statsFeature(alldata, ['idhogar'],'Id','count')

# generate statistical features
#estadocivil_cols = ['estadocivil'+str(i) for i in range(1,8)]
#parentesco_cols = ['parentesco'+str(i) for i in range(1,13)]
#instlevel_cols = ['instlevel'+str(i) for i in range(1,10)]
mean_cols = ['age','rez_esc','escolari']
for col in mean_cols:
    alldata = statsFeature(alldata,['idhogar'],col,'mean')

# new relational features
alldata['bedroom_per_room'] = alldata['bedrooms']/alldata['rooms']
alldata['mobile_per_person'] = alldata['qmobilephone']/alldata['tamviv']
alldata['mobile_per_adult'] = alldata['qmobilephone']/alldata['r4t2']
alldata['tablet_per_adult'] = alldata['v18q1']/alldata['r4t2']
alldata['room_area'] = alldata['hhsize']/alldata['rooms']
alldata['female_ratio'] = alldata['r4m3']/alldata['tamviv']
alldata['male_ratio'] = alldata['r4h3']/alldata['tamviv']
alldata['young_people_ratio'] = alldata['r4t1']/alldata['tamviv']
alldata['young_people_less_than_19_ratio'] = alldata['hogar_nin']/alldata['tamviv']
alldata['old_people_ratio_1'] = alldata['hogar_mayor']/alldata['tamviv']
alldata['old_people_ratio_2'] = alldata['hogar_mayor']/(1+alldata['working_people'])
alldata['adult_ratio'] = alldata['hogar_adul']/alldata['tamviv']
alldata['working_people_ratio'] = alldata['working_people']/alldata['tamviv']
alldata['rent_per_person'] = alldata['v2a1']/alldata['tamviv']
alldata['rent_per_room'] = alldata['v2a1']/alldata['rooms']
alldata['rent_per_working_people'] = alldata['v2a1']/(1+alldata['working_people'])
alldata['rent_area'] = alldata['v2a1']/alldata['hhsize']
alldata['person_area'] = alldata['tamviv']/alldata['hhsize']
alldata['person_per_room'] = alldata['tamviv']/alldata['rooms']
alldata['person_per_bedroom'] = alldata['tamviv']/alldata['bedrooms']
alldata['working_people_area'] = alldata['working_people']/alldata['hhsize']
alldata['working_people_per_room'] = alldata['working_people']/alldata['rooms']

# feature importances
import lightgbm as lgb
# drop id and unnecessary columns
redundant_cols=['Id','agesq','tamviv']+[col for col in alldata.columns if col[:3]=='SQB']
alldata.drop(redundant_cols, axis=1, inplace=True)
# split to train and test data
train_df = alldata[:train_df.shape[0]]
test_df = alldata[-test_df.shape[0]:]
del alldata
gc.collect()

def lgb_f1_macro(preds, dtrain):  
    labels = dtrain.get_label()
    preds = preds.reshape(-1, 4).argmax(axis=1)
    f_score = f1_score(preds, labels, average = 'macro')
    return 'f1_score', f_score, True


lgb_params ={'colsample_bytree': 0.952164731370897, 
                 'scale_pos_weight':1, 
                 'min_child_samples': 111, 
                 'min_child_weight': 0.01, 
                 'num_leaves': 38, 
                 'reg_alpha': 0, 
                 'reg_lambda': 0.1, 
                 'subsample': 0.3029313662262354,
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'metric': 'f1_macro',
                'max_depth': 5,
                'learning_rate': 0.004,
                'bagging_fraction': 0.8, 
                'feature_fraction': 0.9,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 6,
                'lambda_l2': 1.0,
                'min_gain_to_split': 0,
                'num_class': 4,
                'class_weight':'balanced'
            }

X = train_df.values
y = labels-1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2018, stratify=y)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(lgb_params,
                lgb_train,
                num_boost_round=5000,
                feval=lgb_f1_macro,
                valid_sets=[lgb_train, lgb_test],
                feature_name=list(train_df.columns),
                categorical_feature='auto',
                verbose_eval=10)


feature_score = pd.DataFrame(gbm.feature_importance(), columns=['score'])
feature_score['feature'] = train_df.columns
feature_score = feature_score.sort_values(by=['score'], ascending=False)
#print(feature_score)
positive_features = feature_score[feature_score['score']>0]['feature']
# take 80% the best features
feature_names = list(positive_features[-int(len(positive_features)*0.8):])
#print(feature_names)
print(feature_names)
# define f1 score for multiclass 
def xgb_f1_score(y_pred, dtrain):
    y_true = dtrain.get_label()
    return 'f1_macro', f1_score(y_true, y_pred, average='macro')

def xgb_predict(model,X, y, cvFolds=5, early_stopping_rounds=40):
    xgb_param = model.get_xgb_params()
    xgTrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param,
                  xgTrain,
                  num_boost_round=5000,
                  nfold=cvFolds,
                  stratified=True,
                  metrics={'mlogloss'},
                  feval=xgb_f1_score,
                  early_stopping_rounds=early_stopping_rounds,
                  seed=0,     
                  callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])
    model.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm
    model.fit(X, y)
    return model.predict(test_df)

xgb_params = {'n_estimators':20, 
          'max_depth':5,
          'min_child_weight':6,
          'num_class':4,
          'gamma':0.41,
          'learning_rate':0.004,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'objective':'multi:softmax',
          'scale_pos_weight':1,
          'seed':27}

xgb_model = XGBClassifier(**xgb_params)
train_df = train_df[feature_names]
test_df = test_df[feature_names]
pred = xgb_predict(xgb_model, train_df, y)
subm = pd.DataFrame()
subm['Id'] = id_test
subm['Target'] = pred+1
subm.to_csv('submission.csv', index=False)
