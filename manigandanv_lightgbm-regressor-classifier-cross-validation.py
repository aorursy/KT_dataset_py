import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as ssp

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

tfidf = TfidfVectorizer(
	min_df=5, max_features=500, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_text_df["Text"])

test_data = train_text_df.append(test_text_df)
X_tfidf_text = tfidf.transform(test_data["Text"])

#Feature reduction. 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)
SVD_data = svd.fit_transform(X_tfidf_text)

X_train_text = SVD_data [:train_text_df.shape[0]]
X_test_text = SVD_data [train_text_df.shape[0]:]

features = tfidf.get_feature_names()

ID_train = train_variants_df.ID
ID_test = test_variants_df.ID

y = train_variants_df.Class.values-1

train_variants_df = train_variants_df.drop(['ID','Class'], axis=1)
test_variants_df = test_variants_df.drop(['ID'], axis=1)

data = train_variants_df.append(test_variants_df)

X_data = pd.get_dummies(data).values

X = X_data[:train_variants_df.shape[0]]
X_test = X_data[train_variants_df.shape[0]:]

X = ssp.hstack([pd.DataFrame(X_train_text), X], format='csr')
X_test = ssp.hstack((pd.DataFrame(X_test_text), X_test), format='csr')

y_test = np.zeros((X_test.shape[0], max(y)+1))

#LightGBM Regressor
import lightgbm
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
model = LGBMRegressor(boosting_type='gbdt', objective='multiclass',num_class=9,
                      early_stopping = 50,num_iteration=10000,num_leaves=31,is_enable_sparse='true',
                      tree_learner='data',min_data_in_leaf=600,max_depth=4, learning_rate=0.1, 
                      n_estimators=675, max_bin=255, subsample_for_bin=50000, min_split_gain=5, 
                      min_child_weight=5, min_child_samples=10, subsample=0.995, subsample_freq=1, 
                      colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True)

#Fit to training data
model.fit(X, y)
#Generate Predictions
y_pred=model.predict(X_test)
classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame(y_pred, columns=classes)
subm['ID'] = ID_test
#Save predictions to 'output.csv'
subm.to_csv('output.csv', index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as ssp

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

tfidf = TfidfVectorizer(
	min_df=5, max_features=500, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_text_df["Text"])

test_data = train_text_df.append(test_text_df)
X_tfidf_text = tfidf.transform(test_data["Text"])

#Feature reduction. 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)
SVD_data = svd.fit_transform(X_tfidf_text)

X_train_text = SVD_data [:train_text_df.shape[0]]
X_test_text = SVD_data [train_text_df.shape[0]:]

features = tfidf.get_feature_names()

ID_train = train_variants_df.ID
ID_test = test_variants_df.ID

y = train_variants_df.Class.values-1

train_variants_df = train_variants_df.drop(['ID','Class'], axis=1)
test_variants_df = test_variants_df.drop(['ID'], axis=1)

data = train_variants_df.append(test_variants_df)

X_data = pd.get_dummies(data).values

X = X_data[:train_variants_df.shape[0]]
X_test = X_data[train_variants_df.shape[0]:]

X = ssp.hstack([pd.DataFrame(X_train_text), X], format='csr')
X_test = ssp.hstack((pd.DataFrame(X_test_text), X_test), format='csr')

y_test = np.zeros((X_test.shape[0], max(y)+1))

#LightGBM classifier
import lightgbm
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
                       num_class=9,early_stopping = 50,num_iteration=10000,num_leaves=31,
                       is_enable_sparse='true',tree_learner='data',min_data_in_leaf=600,max_depth=4,
                       learning_rate=0.01, n_estimators=675, max_bin=255, subsample_for_bin=50000, 
                       min_split_gain=5, min_child_weight=5, min_child_samples=10, subsample=0.995, 
                       subsample_freq=1, colsample_bytree=1, reg_alpha=0, 
                       reg_lambda=0, seed=0, nthread=-1, silent=True)

#Fit to training data
model.fit(X, y)
#Generate Predictions
y_pred=model.predict_proba(X_test)
classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame(y_pred, columns=classes)
subm['ID'] = ID_test
#Save predictions to 'output.csv'
subm.to_csv('outputclassic1.csv', index=False)
from sklearn import *
import sklearn
import pandas as pd
import numpy as np

train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID')
y = train['Class'].values
train = train.drop('Class', axis=1)

test = pd.merge(test, testx, how='left', on='ID')
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

#commented for Kaggle Limits
'''for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')'''


gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0
#commented for Kaggle Limits
'''for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1'''

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            #('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

y = y - 1 #fix for zero bound array
# LightGBM 
import lightgbm as lgb
import matplotlib.pyplot as plt
lgb_params = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 40, 
    'objective': 'multiclass',
    'num_class':9,
    'tree_learner':'voting',
    'metric':'multi_logloss',
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'max_bin': 100}
# form LightGBM datasets
dtrain_lgb = lgb.Dataset(train, label=y)
# LightGBM, cross-validation
cv_result_lgb = lgb.cv(lgb_params, 
                       dtrain_lgb, 
                       num_boost_round=1000, 
                       nfold=5, 
                       stratified=True, 
                       early_stopping_rounds=50, 
                       verbose_eval=100, 
                       show_stdv=True)
num_boost_rounds_lgb = len(cv_result_lgb['multi_logloss-mean'])
print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
# train model
model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=num_boost_rounds_lgb)
y_pred=model_lgb.predict(test)
classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame(y_pred, columns=classes)
subm['ID'] = pid

subm.to_csv('submission.csv', index=False)