# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import PowerTransformer

import xgboost as xgb

import langid





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import data

train = pd.read_csv('../input/insta_train.csv')

test = pd.read_csv('../input/insta_test.csv')
train.head(3)
train.info()
train.describe()
# #Compare distributions of test and train

# for i in train.select_dtypes(exclude=['object']).drop(['likes'], axis=1):

#     sns.kdeplot(np.log1p(train[i]))

#     sns.kdeplot(np.log1p(test[i]))

#     plt.legend()

#     plt.show() 

 
# #Fix undefined language and target encode language - now too many 'undef' - 7778 from 41540

#1.update language undefined values 

def langupd(df):

    #1.Add a col with lang based on description 

    df['Lang_descr']=df['description'].apply(lambda x: (langid.classify(str(x))[0]) if (abs(langid.classify(str(x))[1])>60) else 'undef')

    #2.Add a col with lang based on caption (for cases when description is NaN)

    df['Lang_capt']=df['caption'].apply(lambda x: (langid.classify(str(x))[0]) if (abs(langid.classify(str(x))[1])>60) else 'undef')

    #3.fill ne column from language for cases when Undefined - use Description lang, when Description Nan - use Caption 

    df['Lang_new']=np.where((df['language']=='undef'),df['Lang_descr'],df['language'])

    df['Lang_new'].where(df['Lang_new']!='undef',df['Lang_capt'], inplace=True)

    df.drop(['Lang_descr', 'Lang_capt', 'language'], inplace=True, axis=1)

    df.rename(columns={'Lang_new':'language'},inplace=True)

    

langupd(train)

langupd(test)
#2 Language target encoding (additive smoothing)

def smoothing_target_encoder(df, column, target, weight=100):

    """

    Target-based encoding is numerization of a categorical variables via the target variable. This replaces the

    categorical variable with just one new numerical variable. Each category or level of the categorical variable

    is represented by it's summary statistic of the target. Main purpose is to deal with high cardinality categorical

    features.

    Smoothing adds the requirement that there must be at least m values for the sample mean to replace the global mean.

    Source: https://www.wikiwand.com/en/Additive_smoothing

    """

    # Compute the global mean

    mean = df[target].mean()

    # Compute the number of values and the mean of each group

    agg = df.groupby(column)[target].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']

    # Compute the 'smoothed' means

    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean

    return df[column].map(smooth)



#train

train['Lang_targ_encoded'] = smoothing_target_encoder(train, column='language', target='likes')

lang_dict = train.set_index('language').to_dict()['Lang_targ_encoded']

#add languages from test(absent in train)

lang_dict['wa']=lang_dict['pt']

lang_dict['te']=lang_dict['en']



#test

test['Lang_targ_encoded']=test['language'].replace(lang_dict)

test['Lang_targ_encoded'].astype('float64')





##encode userlanguage

#train

train['userlang']=train.groupby(['user'])['language'].transform(lambda x: pd.Series.mode(x)[0])

train['userlang_encoded'] = smoothing_target_encoder(train, column='userlang', target='likes')

user_lang_dict = train.set_index('userlang').to_dict()['userlang_encoded']

#add languages from test(absent in train)

user_lang_dict['wa']=user_lang_dict['pt']

user_lang_dict['te']=user_lang_dict['en']

user_lang_dict['rw']=user_lang_dict['undef']

user_lang_dict['hr']=user_lang_dict['bs']

user_lang_dict['sq']=user_lang_dict['mt']

user_lang_dict['co']=user_lang_dict['undef']



#test

test['userlang']=test.groupby(['user'])['language'].transform(lambda x: pd.Series.mode(x)[0])

test['userlang_encoded']=test['userlang'].replace(user_lang_dict).astype('float64')
# PolynomialFeatures exploration

# from sklearn.preprocessing import PolynomialFeatures

# poly_features = PolynomialFeatures(degree=2) 

# X_poly_train = poly_features.fit(X_train)

# feature_names=X_poly_train.get_feature_names(X_train.columns)

# X_poly_train = poly_features.fit_transform(X_train)

# X_poly_test = poly_features.fit_transform(X_test)

# test_x_poly=poly_features.fit_transform(test_x)



#After check feature importance following columns will be added

#'comments followers'

#'followers comments_mean'

#'followers Lang_targ_encoded'

#'fol pos'

#'followers^2'
def transform(df):

    df['img_len'] = df['img'].str.len()

    df['description_len'] = df['description'].str.len()

    df['caption_upper'] = df['caption'].str.findall(r'[A-Z]').str.len()

    df['po_co'] = df['posts']/(df['comments']+1)

    df['pof'] = df['posts']/(df['followings']+1)

    df['user_count'] = df.groupby('user')['posts'].transform('count')

    df['username_len'] = df['username'].str.len()

    df['fol'] = df['followers']/(df['followings']+1)

    df['act'] = df['comments']/(df['followers']+1)

    df['pos'] = df['posts']/(df['followers']+1)

    df['reference_mean'] = df.groupby('user')['reference'].transform('mean')

    df['reference_max'] = df.groupby('user')['reference'].transform('max')

    df['hashtag_max'] = df.groupby('user')['hashtag'].transform('max')

    df['hashtag_min'] = df.groupby('user')['hashtag'].transform('min')

    df['hashtag_mean'] = df.groupby('user')['hashtag'].transform('mean')

    df['hashtag_std'] = df.groupby('user')['hashtag'].transform('std')

    df['comments_max'] = df.groupby('user')['comments'].transform('max')

    df['comments_min'] = df.groupby('user')['comments'].transform('min')

    df['comments_mean'] = df.groupby('user')['comments'].transform('mean')

    df['comments_std'] = df.groupby('user')['comments'].transform('std')

    df['comments_followers']=df['comments']*df['followers']

    df['followers_comments_mean']=df['comments_mean']*df['followers']

    df['followers_Lang_targ_encoded']=df['Lang_targ_encoded']*df['followers']

    df['fol2']=df['fol']*df['fol']

    df['fol_pos']=df['fol']*df['pos']

    df['fol_pow'] = df['followers']*(df['followings'])

    df['po_co_pow'] = df['posts']*(df['comments'])

    df['description_len'] = df['description'].str.len()

    df['caption_upper'] = df['caption'].str.findall(r'[A-Z]').str.len()

    df['comments_mean_diff'] = df['comments'] / (df['comments_mean']+1)

    df['bot'] = df['user'].str.count('_|#|\.|,').astype(int)

    df['bot_caption'] = pd.to_numeric(df['caption'].str.count('_|#|\.|,'), errors='coerce')

    df['numbers_user'] = df['user'].str.contains('0|1|2|3|4|5|6|7|8|9').astype(int)

    df['numbers_caption'] = pd.to_numeric(df['caption'].str.contains('0|1|2|3|4|5|6|7|8|9'), errors='coerce')

    df['s1080.1080'] = pd.to_numeric(df['img'].str.contains('1080\.1080'), errors='coerce')

    df['e35'] = pd.to_numeric(df['img'].str.contains('e35'), errors='coerce')



    return df
#Feature correlation

corr_matrix=train.corr().abs()#[df.corr()>0.5] to remove diagonal

fig, ax = plt.subplots(figsize=(8,8))  

sns.heatmap(corr_matrix, 

            xticklabels=corr_matrix.columns.values,

            yticklabels=corr_matrix.columns.values,

            cmap=sns.diverging_palette(20, 220, n=200),

            square=True,

            ax=ax)
train.boxplot(column='likes', by = 'user')
#Transform and assign x,y /XGBoost has an in-built routine to handle missing values)

X = transform(train).select_dtypes(exclude=['object']).drop(['likes'], axis=1)

test_x = transform(test).select_dtypes(exclude=['object'])

y = train['likes']



#split data

X_train, X_test, y_train_init, y_test_init = train_test_split(X, y, test_size=0.1, random_state=42)





#Log Y. Exponentiate prediction back to Y. e^W will always be positive

y_train=np.log1p(y_train_init)

y_test=np.log1p(y_test_init)

y=np.log1p(y)



#to use the native API for XGBoost, we need to build DMatrices

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

#dtrain full for cross val:

dtrain_full=xgb.DMatrix(X, label=np.log1p(y))
# #Grid search

# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer

# model = xgb.XGBRegressor(objective='mae',

#                          booster='gbtree', 

#                          base_score=0.5, 

#                          random_state=42,

#                          #eval_metric='mae',

#                          max_depth=9,

#                          learning_rate=0.1, 

#                          n_estimators=200, 

#                         n_jobs=-1, 

#                         min_child_weight=5, 

#                         subsample=0.8, 

#                         reg_alpha=0,

#                         reg_lambda=3   

#                         )



# def smape(A, F):

#     A = 10**(A)-1

#     F = 10**(F)-1

#     return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



# scoring = {'smape': make_scorer(smape, greater_is_better=False)}







# clf = GridSearchCV(model,{'max_depth': [5,7,9],

#                           #'min_child_weight':[7],

#                           #'n_estimators': [200,300,400,500],

#                           #'learning_rate':[0.1],

#                           #'reg_alpha': [0.2],

#                           #'reg_lambda':[3], 

#                           #'subsample':[0.8],

#                           #'colsample_bytree':[0.8]  

#                          },

#                    verbose=,

#                    scoring=scoring

#                    cv=5,

#                    n_jobs=-1

#                   )



# clf.fit(X, y,groups = np.concatenate([np.repeat(0, 8308), np.repeat(1, 8308), np.repeat(2, 8308), np.repeat(3, 8308), np.repeat(4, 8308)]))

# print('\n Best score:')

# print(clf.best_score_)

# print('\n Best params:')

# print(clf.best_params_)

# print('\n Best estimator:')

# print(clf.best_estimator_)

#Best model

model_xgb = xgb.XGBRegressor(max_depth=7, 

                             learning_rate=0.01, 

                             n_estimators=870,  

                             booster='gbtree', 

                             n_jobs=-1, 

                             min_child_weight=5, 

                             subsample=0.8, 

                             colsample_bytree=0.8, 

                             base_score=0.5, 

                             random_state=42,

                             reg_alpha=0.2,

                             reg_lambda=3

                            )

model_xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='mae')

y_pred = model_xgb.predict(X_test)



y_pred2= np.expm1(y_pred)

y_test2= np.expm1(y_test)



RMSE = mean_squared_error(y_test2, y_pred2)**0.5

MAE=mean_absolute_error(y_test2, y_pred2)

SMAPE=(200/y_pred2.size)*sum(abs(y_pred2-y_test2)/(abs(y_pred2)+abs(y_test2)))

print('RMSE={}, MAE={}, SMAPE={}'.format(RMSE, MAE, SMAPE))



model_xgb.score(X_test, y_test)
#Predict vs real - scatter

ax=plt.scatter(y_pred, y_test)

plt.title('Predicted number of likes vs true number of liks')

plt.ylabel('True number of likes') 

plt.xlabel('Predicted number of likes')

fig = ax.figure

fig.set_size_inches(8, 10)
# #Test vs residuals

# ax=plt.scatter(y_test, abs(y_pred-y_test))

# plt.title('Residuals of predicted number of likes vs true number of likes')

# plt.xlabel('True number of likes') 

# plt.ylabel('Residuals predicted number of likes')

# fig = ax.figure

# fig.set_size_inches(8, 10)
#Feature importance

feature_imp = pd.DataFrame(sorted(zip(model_xgb.feature_importances_, X.columns.astype('str'))), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))

plt.tight_layout()

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.show()


# prediction = pd.read_csv('sample_submission.csv')

# prediction['likes'] = np.expm1(model_xgb.predict(test_x))#model_xgb.predict(test_x)



# prediction.to_csv('my_submission.csv', index=False)