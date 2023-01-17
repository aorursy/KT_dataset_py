# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/used-electronics-data/Train.csv")

print(train.shape)

test = pd.read_csv("/kaggle/input/used-electronics-data/Test.csv")

sample_sub = pd.read_excel('/kaggle/input/used-electronics-data/Sample_Submission.xlsx')
# train = train[(train.State.isin(test.State)) & (train.City.isin(test.City))]
train.Price.describe().T
train[train.Price>=100000]
train[train['Locality']==955]
test[test['Locality']==955]
from wordcloud import WordCloud, STOPWORDS

stopwords1 = set(STOPWORDS)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.utils import shuffle



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords1,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(train[train.Price>=100000]['Model_Info'])
show_wordcloud(train[train.Price<=12000.000000]['Model_Info'])
import re

txt = "name3333 sdcsefew"

re.sub(r"name[0-9]+", '', txt)
def find(v):

    if v.find("32gb")!=-1 or v.find("32 gb")!=-1:

        return 1

    elif v.find("64gb")!=-1 or v.find("64 gb")!=-1:

        return 2

    elif v.find("128gb")!=-1 or v.find("128 gb")!=-1:

        return 3

    elif v.find("256gb")!=-1 or v.find("256 gb")!=-1:

        return 4

    else:

        return 0

    
train['Model_Info'].values
train['Brand'].nunique()
df=train.append(test,ignore_index=True)

df.head()
# df['class2']=df['Model_Info'].apply(find1)
# df['model_len']

df['Class'] = df['Model_Info'].apply(find)



df['Model_Info']
df['name_col']=df['Model_Info'].apply(lambda x: " ".join([y for y in x.split() if y.startswith('name')]))
# df['Model_Info'] = df['Model_Info'].apply(rep)
# df['gb_col']=df['Model_Info'].apply(lambda x: x.count('gb'))
from sklearn.preprocessing import LabelEncoder

l=LabelEncoder()

df['name_col']=l.fit_transform(df['name_col'])



# df['gb_col']=l.fit_transform(df['gb_col'])
df['Class'].value_counts()
df['Model_Info']
# df['Model_Info'].apply(lambda x: " ".join([y for y in x.split() if y.endswith('gb')]))
# df['Additional_Description_len'] = df['Additional_Description'].apply(lambda x: x.count('10100'))
# df.info()
df=pd.get_dummies(df,columns=['State','City','Brand'],drop_first=True)

df.head()
# df['name_col']=df['Model_Info'].apply(lambda x: x.count('name'))
feats = [x for x in df.columns if x not in ['Model_Info','Additional_Description','Price','Locality']]
dftrain, dftest = df[df.Price.isnull()==False], df[df.Price.isnull()==True]

dftest.reset_index(drop=True, inplace=True)

target = np.log1p(dftrain['Price'])
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

v_1 = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word')

typ_tr =v_1.fit_transform(dftrain['Model_Info'])

typ_ts =v_1.transform(dftest['Model_Info'])





v_1c = TfidfVectorizer(ngram_range=(2,6),stop_words="english", analyzer='char')

typ_trc =v_1c.fit_transform(dftrain['Model_Info'])

typ_tsc =v_1c.transform(dftest['Model_Info'])



# v_2c = TfidfVectorizer(ngram_range=(1,4),stop_words="english", analyzer='char')

# typ_trc2 =v_2c.fit_transform(dftrain['Model_Info'])

# typ_tsc2 =v_2c.transform(dftest['Model_Info'])





# v_2 = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

# res_tr =v_2.fit_transform(dftrain['Additional_Description'])

# res_ts =v_2.transform(dftest['Additional_Description'])
# v_1.vocabulary_

dftrain[feats]


from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((dftrain[feats],typ_tr,typ_trc)).tocsr()

final_featurest = sparse.hstack((dftest[feats],typ_ts ,typ_tsc)).tocsr()
final_features
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss,mean_squared_log_error



X_trn, X_val, y_trn, y_val = train_test_split(final_features, target, test_size=0.25, random_state=1994)

X_test = final_featurest
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor



# clf = LGBMClassifier(learning_rate=0.05, colsample_bytree=0.3, reg_alpha=3, reg_lambda=3, max_depth=-1, n_estimators=2000, min_child_samples=15, num_leaves=141)

clf = XGBRegressor(n_estimators=2000,learning_rate=0.1,colsample_bytree=0.5,random_state=1994,min_child_samples=2)

clf.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], verbose=100, early_stopping_rounds=100)

predictions_val_lgb = clf.predict(X_val)

print(f"RMSLE is: {np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(predictions_val_lgb)))}")
from lightgbm import LGBMRegressor



# clf = LGBMClassifier(learning_rate=0.05, colsample_bytree=0.3, reg_alpha=3, reg_lambda=3, max_depth=-1, n_estimators=2000, min_child_samples=15, num_leaves=141)

clf = LGBMRegressor(learning_rate=0.1, n_estimators=4000,colsample_bytree=0.5, reg_alpha=0.5, min_child_samples=2, num_leaves=30)

_ = clf.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], verbose=100, early_stopping_rounds=100,eval_metric='RMSE')

predictions_val_lgb = clf.predict(X_val)

print(f"RMSLE is: {np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(predictions_val_lgb)))}")
X=final_features

y=target
y_pred_tot=[]

err=[]

feature_importance_df = pd.DataFrame()



from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=15,shuffle=True,random_state=1994)

i=1

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=LGBMRegressor(learning_rate=0.1, n_estimators=4000,colsample_bytree=0.5, reg_alpha=0.5, min_child_samples=2, num_leaves=30)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200,eval_metric='RMSE')

    preds=m.predict(X_test,num_iteration=m.best_iteration_)

    print(f"RMSLE is: {np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(preds)))}")

    err.append(np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(preds))))

    p = m.predict(final_featurest)

    i=i+1

    y_pred_tot.append(p)
y_pred_totxgb=[]

errx=[]

feature_importance_df = pd.DataFrame()



from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=15,shuffle=True,random_state=1994)

i=1

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=XGBRegressor(n_estimators=2000,learning_rate=0.1,colsample_bytree=0.5,random_state=1994,min_child_samples=2)

    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=200,verbose=200)

    preds=m.predict(X_test,ntree_limit=m.best_iteration)

    print(f"RMSLE is: {np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(preds)))}")

    errx.append(np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(preds))))

    p = m.predict(final_featurest)

    i=i+1

    y_pred_totxgb.append(p)
np.mean(err),np.mean(errx)

# np.expm1(np.mean(y_pred_tot,0))*0.6+np.expm1(np.mean(y_pred_totxgb,0))*0.4
sample_sub['Price']=np.expm1(np.mean(y_pred_tot,0))*0.5+np.expm1(np.mean(y_pred_totxgb,0))*0.5
sample_sub.to_excel('kv11_ensemble.xlsx',index=False)
sample_sub['Price']=np.expm1(np.mean(y_pred_totxgb,0))

sample_sub.to_excel('kv11_xgb.xlsx',index=False)
sample_sub['Price']=np.expm1(np.mean(y_pred_tot,0))

sample_sub.to_excel('kv11_lgb.xlsx',index=False)