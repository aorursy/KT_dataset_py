import numpy as np
import string
import re
from sklearn import pipeline
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn import metrics, cross_validation
from catboost import CatBoostClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
warnings.filterwarnings('ignore')
sns.set()
%matplotlib inline
df=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
doub = df_test.groupby(['Word'])['Word'].count()
df_test = df_test.join(doub, on='Word', rsuffix='_d')
li = df_test[df_test.Word_d == 2].index
limi = li[::2]
li = li[1::2]
df_test = df_test.drop('Word_d', axis=1)
def MyPipe(model, scaler):
    pipe = pipeline.Pipeline([
                              ('vect', CountVectorizer(ngram_range=(1,7), analyzer='char', lowercase=False)),
                              ('scl', scaler),
                              ('clf', model)
                            ])
    return pipe
def MyPipe2(model, scaler):
    pipe = pipeline.Pipeline([
                              ('scl', scaler),
                              ('clf', model)
                            ])
    return pipe
cv = StratifiedKFold(df.Label, n_folds=5, shuffle=True)
df['char_count']=df.Word.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df['1end'] = df['Word'].str[-1:].str.lower()
df_test['1end'] = df_test['Word'].str[-1:].str.lower()
df['2end'] = df['Word'].str[-2:].str.lower()
df_test['2end'] = df_test['Word'].str[-2:].str.lower()
df['2first'] = df['Word'].str[:2].str.lower()
df_test['2first'] = df_test['Word'].str[:2].str.lower()
df['1first'] = df['Word'].str[:1].str.lower()
df_test['1first'] = df_test['Word'].str[:1].str.lower()
df['3first'] = df['Word'].str[:3].str.lower()
df_test['3first'] = df_test['Word'].str[:3].str.lower()
df['3end'] = df['Word'].str[-3:].str.lower()
df_test['3end'] = df_test['Word'].str[-3:].str.lower()
df_test['char_count']=df_test.Word.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df['title_count'] = df.Word.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df_test['title_count'] = df_test.Word.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df['is_upper']=df.Word.map(lambda x:1 if x.isupper() else 0)
df_test['is_upper']=df_test.Word.map(lambda x:1 if x.isupper() else 0)
df["count_punctuations"] =df["Word"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_test["count_punctuations"] =df_test["Word"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df["count_words_upper"] = df["Word"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df_test["count_words_upper"] = df_test["Word"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df['Vow'] = df['Word'].apply(lambda x: len(re.findall('[уеыаоэяию]', x, re.IGNORECASE)))
df_test['Vow'] = df_test['Word'].apply(lambda x: len(re.findall('[уеыаоэяию]', x, re.IGNORECASE)))
df['noVow'] = df['char_count']-df['Vow']
df_test['noVow'] = df_test['char_count']-df_test['Vow']
df["has_digit"]=df.Word.map(lambda x: 1 if len([int(s) for s in set(x) if s.isdigit()])>0 else 0)
df_test["has_digit"]=df_test.Word.map(lambda x: 1 if len([int(s) for s in set(x) if s.isdigit()])>0 else 0)
train_col=['Word']
vect_char = CountVectorizer(lowercase=False, analyzer='char',
                        ngram_range=(1,7))
tf_char = vect_char.fit(list(df[train_col[0]]))
tr_vect = tf_char.transform(df[train_col[0]])
tr_vect_test = tf_char.transform(df_test[train_col[0]])
t_duble=df.Word.value_counts().to_dict()
te_duble=df_test.Word.value_counts().to_dict()
df['is_dublicate']=df.Word.map(lambda x:1 if t_duble[x]>1 else 0)
df_test['is_dublicate']=df_test.Word.map(lambda x:1 if te_duble[x]>1 else 0)
df['num_duble']=0
df_test['num_duble']=0
dublicate=0
for i, item in enumerate(df['Word']):
    if (df.iloc[i,8]==1) and (dublicate==0):
        dublicate=1
        df.iloc[i,9]=1
        continue
    if (df.iloc[i,8]==1) and (dublicate==1):
        dublicate=0
        df.iloc[i,9]=2
dublicate=0
for i, item in enumerate(df_test['Word']):
    if (df_test.iloc[i,8]==1) and (dublicate==0):
        dublicate=1
        df_test.iloc[i,9]=1
        continue
    if (df_test.iloc[i,8]==1) and (dublicate==1):
        dublicate=0
        df_test.iloc[i,9]=2
#Для желающих опробовать pymorphy+XGB+CB
"""import pymorphy2
morph = pymorphy2.MorphAnalyzer()
all_data = pd.concat([df , df_test])
all_data['pymorphy'] = all_data['Word'].apply(lambda x: morph.tag(x)[0])

all_data['pymorphy_animacy'] = all_data['pymorphy'].apply(lambda x: x.animacy)
all_data['pymorphy_POS'] = all_data['pymorphy'].apply(lambda x: x.POS)
all_data['pymorphy_case'] = all_data['pymorphy'].apply(lambda x: x.case)
all_data['pymorphy_number'] = all_data['pymorphy'].apply(lambda x: x.number)
all_data['pymorphy_gender'] = all_data['pymorphy'].apply(lambda x: x.gender)

all_data.drop('pymorphy' , axis=1 , inplace=True)

columns_to_one_hot = ['pymorphy_animacy', 'pymorphy_POS', 'pymorphy_case','pymorphy_number', 'pymorphy_gender']
for col in columns_to_one_hot:
    all_data[col] = LabelEncoder().fit_transform(list(all_data[col].fillna('nan')))

new_train = all_data[all_data['Label'].notnull()]
new_test = all_data[all_data['Label'].isnull()]"""
xgb_col=['1end', '1first', '2end', '2first', '3end', '3first', 'num_duble', 'Vow', 'char_count', 'count_punctuations', 'count_words_upper',
       'has_digit', 'is_dublicate', 'is_upper', 'title_count']
float_col=['num_duble', 'Vow', 'char_count', 'count_punctuations', 'count_words_upper',  'noVow',
       'has_digit', 'is_dublicate', 'is_upper', 'title_count']
from scipy import sparse
X = sparse.hstack([
                   tr_vect,
                   #new_train[xgb_col]
                   df[float_col]
])
x_test = sparse.hstack([
                        tr_vect_test,
                        #new_test[xgb_col]
                    df_test[float_col]
])
log_reg = LogisticRegression(penalty='l2', n_jobs=4, random_state=42,
                             class_weight='balanced',
                             C = 0.1, max_iter=50, solver='sag')
my_pipe_log=MyPipe2(log_reg, MaxAbsScaler())
score_log=cross_val_score(my_pipe_log, X, df.Label, cv=cv, scoring='roc_auc')
print("ROC-AUC (LR): {}/{}".format(score_log.mean(), score_log.std()))
my_pipe_log.fit(X, df.Label)
svm_sgd=SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=150, random_state=42, 
                      n_jobs=4, class_weight='balanced'
                     )
sgd_pipe=MyPipe2(svm_sgd, MaxAbsScaler())
score_svm_sgd=cross_val_score(sgd_pipe, X, df.Label, cv=cv, n_jobs=4, scoring='roc_auc')
print("ROC-AUC (LR): {}/{}".format(score_svm_sgd.mean(), score_svm_sgd.std()))
sgd_pipe.fit(X, df.Label)

cb=CatBoostClassifier(iterations=100,  thread_count=4, verbose=False)
cat_col=[0,1,2,3,4,5]
score_rf=cross_val_score(cb,
                         df[xgb_col], df.Label, 
                         fit_params={'cat_features': cat_col},
                         cv=cv, n_jobs=4, scoring='roc_auc')
print("auc (RF): {}/{}".format(round(score_rf.mean(),4), round(score_rf.std(),4)))
from sklearn.model_selection import train_test_split
X_t, X_te, y_t, y_te=train_test_split(df[xgb_col], df.Label, stratify=df.Label, 
                                                  random_state=42, test_size=0.25)
model_cb=CatBoostClassifier(iterations=20, depth=15,
                        learning_rate=0.5,
                        random_seed=42, thread_count=4,
                        rsm=1, 
                        l2_leaf_reg=2,
                        loss_function='Logloss')
model_cb.fit(df[xgb_col], df.Label, cat_features=cat_col)
#eclf_pred=clf_ensmbl.predict_proba(x_test)[:,1]
#log_pred=my_pipe_log.predict_proba(x_test)[:,1]
sgd_pred=sgd_pipe.predict_proba(x_test)[:,1]
#cb_pred=model_cb.predict_proba(new_test[xgb_col])[:,1]
df_test['proba']=sgd_pred#sgd_pred #log_pred*0.2+sgd_pred*0.8
ma = df_test['proba'].max()
mi = df_test['proba'].min()
df_test['proba'].loc[li] = ma
df_test['proba'].loc[limi] = mi

preds=pd.DataFrame({'Id':range(len(df_test)),'Prediction':df_test['proba']})
preds.to_csv('23022018-2.csv', sep=',',  index=False)
