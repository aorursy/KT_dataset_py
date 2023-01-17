import pylab as pl # linear algebra + plots
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gc
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, Counter
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from scipy.stats import pearsonr
from scipy.sparse import hstack
from multiprocessing import Pool

Folder = "../input/"
Ttr = pd.read_csv(Folder + 'train.csv')
Tts = pd.read_csv(Folder + 'test.csv', low_memory=False)
R = pd.read_csv(Folder + 'resources.csv')
# combine the tables into one
target = 'project_is_approved'
Ttr['tr'] = 1; Tts['tr'] = 0
Ttr['ts'] = 0; Tts['ts'] = 1

T = pd.concat((Ttr,Tts))

T.loc[T.project_essay_4.isnull(), ['project_essay_4','project_essay_2']] = \
    T.loc[T.project_essay_4.isnull(), ['project_essay_2','project_essay_4']].values

T[['project_essay_2','project_essay_3']] = T[['project_essay_2','project_essay_3']].fillna('')

T['project_essay_1'] = T.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                                     str(row['project_essay_2'])]), axis=1)
T['project_essay_2'] = T.apply(lambda row: ' '.join([str(row['project_essay_3']),
                                                     str(row['project_essay_4'])]), axis=1)

T = T.drop(['project_essay_3', 'project_essay_4'], axis=1)

R['priceAll'] = R['quantity']*R['price']
newR = R.groupby('id').agg({'description':'count',
                            'quantity':'sum',
                            'price':'sum',
                            'priceAll':'sum'}).rename(columns={'description':'items'})
newR['avgPrice'] = newR.priceAll / newR.quantity
numFeatures = ['items', 'quantity', 'price', 'priceAll', 'avgPrice']

for func in ['min', 'max', 'mean']:
    newR = newR.join(R.groupby('id').agg({'quantity':func,
                                          'price':func,
                                          'priceAll':func}).rename(
                                columns={'quantity':func+'Quantity',
                                         'price':func+'Price',
                                         'priceAll':func+'PriceAll'}).fillna(0))
    numFeatures += [func+'Quantity', func+'Price', func+'PriceAll']

newR = newR.join(R.groupby('id').agg(
    {'description':lambda x:' '.join(x.values.astype(str))}).rename(
    columns={'description':'resource_description'}))

T = T.join(newR, on='id')

# if you visit the donors website, it has categorized the price by these bins:
T['price_category'] = pl.digitize(T.priceAll, [0, 50, 100, 250, 500, 1000, pl.inf])
numFeatures.append('price_category')
# the difference of max and min of price and quantity per item can also be relevant
for c in ['Quantity', 'Price', 'PriceAll']:
    T['max%s_min%s'%(c,c)] = T['max%s'%c] - T['min%s'%c]
    numFeatures.append('max%s_min%s'%(c,c))

del Ttr, Tts, R, newR
gc.collect();
le = LabelEncoder()
T['teacher_id'] = le.fit_transform(T['teacher_id'])
T['teacher_gender_unknown'] = T.teacher_prefix.apply(lambda x:int(x not in ['Ms.', 'Mrs.', 'Mr.']))
numFeatures += ['teacher_number_of_previously_posted_projects','teacher_id','teacher_gender_unknown']

statFeatures = []
for col in ['school_state', 'teacher_id', 'teacher_prefix', 'teacher_gender_unknown', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_number_of_previously_posted_projects']:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')
%%time
textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']

def getSentFeat(s):
    sent = TextBlob(s).sentiment
    return (sent.polarity, sent.subjectivity)

print('sentimental analysis')
with Pool(4) as p:
    for col in textColumns:
        temp = pl.array(list(p.map(getSentFeat, T[col])))
        T[col+'_pol'] = temp[:,0]
        T[col+'_sub'] = temp[:,1]
        numFeatures += [col+'_pol', col+'_sub']

print('key words')
KeyChars = ['!', '\?', '@', '#', '\$', '%', '&', '\*', '\(', '\[', '\{', '\|', '-', '_', '=', '\+',
            '\.', ':', ';', ',', '/', '\\\\r', '\\\\t', '\\"', '\.\.\.', 'etc', 'http', 'poor',
            'military', 'traditional', 'charter', 'head start', 'magnet', 'year-round', 'alternative',
            'art', 'book', 'basics', 'computer', 'laptop', 'tablet', 'kit', 'game', 'seat',
            'food', 'cloth', 'hygiene', 'instraction', 'technolog', 'lab', 'equipment',
            'music', 'instrument', 'nook', 'desk', 'storage', 'sport', 'exercise', 'trip', 'visitor',
            'my students', 'our students', 'my class', 'our class']
for col in textColumns:
    for c in KeyChars:
        T[col+'_'+c] = T[col].apply(lambda x: len(re.findall(c, x.lower())))
        numFeatures.append(col+'_'+c)

#####
print('num words')
for col in textColumns:
    T['n_'+col] = T[col].apply(lambda x: len(x.split()))
    numFeatures.append('n_'+col)
    T['nUpper_'+col] = T[col].apply(lambda x: sum([s.isupper() for s in list(x)]))
    numFeatures.append('nUpper_'+col)

#####
print('word tags')
Tags = ['CC', 'CD', 'DT', 'IN', 'JJ', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 
        'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 
        'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
def getTagFeat(s):
    d = Counter([t[1] for t in pos_tag(s.split())])
    return [d[t] for t in Tags]

with Pool(4) as p:
    for col in textColumns:
        temp = pl.array(list(p.map(getTagFeat, T[col])))
        for i, t in enumerate(Tags):
            if temp[:,i].sum() == 0:
                continue
            T[col+'_'+t] = temp[:, i]
            numFeatures += [col+'_'+t]

#####
print('common words')
for i, col1 in enumerate(textColumns[:-1]):
    for col2 in textColumns[i+1:]:
        T['%s_%s_common' % (col1, col2)] = T.apply(lambda row:len(set(re.split('\W', row[col1].lower())).intersection(re.split('\W', row[col2].lower()))), axis=1)
        numFeatures.append('%s_%s_common' % (col1, col2))


pl.figure(figsize=(15,5))
sns.violinplot(data=T,x=target,y='project_essay_2_!');
pl.figure(figsize=(15,5))
sns.violinplot(data=T,x=target,y='project_essay_1_!');
dateCol = 'project_submitted_datetime'
def getTimeFeatures(T):
    T['year'] = T[dateCol].apply(lambda x: x.year)
    T['month'] = T[dateCol].apply(lambda x: x.month)
    T['day'] = T[dateCol].apply(lambda x: x.day)
    T['dow'] = T[dateCol].apply(lambda x: x.dayofweek)
    T['hour'] = T[dateCol].apply(lambda x: x.hour)
    T['days'] = (T[dateCol]-T[dateCol].min()).apply(lambda x: x.days)
    return T

T[dateCol] = pd.to_datetime(T[dateCol])
T = getTimeFeatures(T)

P_tar = T[T.tr==1][target].mean()
timeFeatures = ['year', 'month', 'day', 'dow', 'hour', 'days']
for col in timeFeatures:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')

numFeatures += timeFeatures
numFeatures += statFeatures
%%time
T2 = T[numFeatures+['id','tr','ts',target]].copy()
Ttr = T2[T.tr==1]
Tar_tr = Ttr[target].values
n = 10
inx = [pl.randint(0, Ttr.shape[0], int(Ttr.shape[0]/n)) for k in range(n)]
# inx is used for crossvalidation of calculating the correlation and p-value
Corr = {}
for c in numFeatures:
    # since some values might be 0s, I use x+1 to avoid missing some important relations
    C1,P1=pl.nanmean([pearsonr(Tar_tr[inx[k]],   (1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)
    C2,P2=pl.nanmean([pearsonr(Tar_tr[inx[k]], 1/(1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)
    if P2<P1:
        T2[c] = 1/(1+T2[c])
        Corr[c] = [C2,P2]
    else:
        T2[c] = 1+T2[c]
        Corr[c] = [C1,P1]

polyCol = []
thrP = 0.01
thrC = 0.02
print('columns \t\t\t Corr1 \t\t Corr2 \t\t Corr Combined')
for i, c1 in enumerate(numFeatures[:-1]):
    C1, P1 = Corr[c1]
    for c2 in numFeatures[i+1:]:
        C2, P2 = Corr[c2]
        V = T2[c1] * T2[c2]
        Vtr = V[T2.tr==1].values
        C, P = pl.nanmean([pearsonr(Tar_tr[inx[k]], Vtr[inx[k]]) for k in range(n)], 0)
        if P<thrP and abs(C) - max(abs(C1),abs(C2)) > thrC:
            T[c1+'_'+c2+'_poly'] = V
            polyCol.append(c1+'_'+c2+'_poly')
            print(c1+'_'+c2, '\t\t(%g, %g)\t(%g, %g)\t(%g, %g)'%(C1,P1, C2,P2, C,P))

numFeatures += polyCol
print(len(numFeatures))
del T2, Ttr
gc.collect();
pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='maxPrice')
pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='meanPrice')
pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='maxPrice_meanPrice_poly');
def getCatFeatures(T, Col):
    vectorizer = CountVectorizer(binary=True,
                                 ngram_range=(1,1),
                                 tokenizer=lambda x:[a.strip() for a in x.split(',')])
    return vectorizer.fit_transform(T[Col].fillna(''))

X_tp = getCatFeatures(T, 'teacher_prefix')
X_ss = getCatFeatures(T, 'school_state')
X_pgc = getCatFeatures(T, 'project_grade_category')
X_psc = getCatFeatures(T, 'project_subject_categories')
X_pssc = getCatFeatures(T, 'project_subject_subcategories')

X_cat = hstack((X_tp, X_ss, X_pgc, X_psc, X_pssc))

del X_tp, X_ss, X_pgc, X_psc, X_pssc
%%time
# from nltk.stem.wordnet import WordNetLemmatizer
# from autocorrect import spell  # as spell checker and corrector
# L = WordNetLemmatizer()
p = PorterStemmer()
def wordPreProcess(sentence):
    return ' '.join([p.stem(x.lower()) for x in re.split('\W', sentence) if len(x) >= 1])
# return ' '.join([p.stem(L.lemmatize(spell(x.lower()))) for x in re.split('\W', sentence) if len(x) > 1])


def getTextFeatures(T, Col, max_features=10000, ngrams=(1,2), verbose=True):
    if verbose:
        print('processing: ', Col)
    vectorizer = CountVectorizer(stop_words=None,
                                 preprocessor=wordPreProcess,
                                 max_features=max_features,
                                 binary=True,
                                 ngram_range=ngrams)
#     vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
#                                  preprocessor=wordPreProcess,
#                                  max_features=max_features)
    X = vectorizer.fit_transform(T[Col])
    return X, vectorizer.get_feature_names()

n_es1, n_es2, n_prs, n_rd, n_pt = 3000, 8000, 2000, 3000, 1000
X_es1, feat_es1 = getTextFeatures(T, 'project_essay_1', max_features=n_es1)
X_es2, feat_es2 = getTextFeatures(T, 'project_essay_2', max_features=n_es2)
X_prs, feat_prs = getTextFeatures(T, 'project_resource_summary', max_features=n_prs)
X_rd, feat_rd = getTextFeatures(T, 'resource_description', max_features=n_rd, ngrams=(1,3))
X_pt, feat_pt = getTextFeatures(T, 'project_title', max_features=n_pt)

X_txt = hstack((X_es1, X_es2, X_prs, X_rd, X_pt))
del X_es1, X_es2, X_prs, X_rd, X_pt

# 
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(1000)
# X_txt = svd.fit_transform(X_txt)
from sklearn.preprocessing import StandardScaler
X = hstack((X_txt, X_cat, StandardScaler().fit_transform(T[numFeatures].fillna(0)))).tocsr()

Xtr = X[pl.find(T.tr==1), :]
Xts = X[pl.find(T.ts==1), :]
Ttr_tar = T[T.tr==1][target].values
Tts = T[T.ts==1][['id',target]]

Yts = []
del T
del X
gc.collect();

from keras.layers import Input, Dense, Flatten, concatenate, Dropout, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def breakInput(X1):
    X2 = []
    i = 0
    for n in [n_es1, n_es2, n_prs, n_rd, n_pt, X_cat.shape[1], len(numFeatures)]:
        X2.append(X1[:,i:i+n])
        i += n
    return X2

def getModel(HLs, Drop=0.25, OP=optimizers.Adam()):
    temp = []
    inputs_txt = []
    for n in [n_es1, n_es2, n_prs, n_rd, n_pt]:
        input_txt = Input((n, ))
        X_feat = Dropout(Drop)(input_txt)
        X_feat = Dense(int(n/100), activation="linear")(X_feat)
        X_feat = Dropout(Drop)(X_feat)
        temp.append(X_feat)
        inputs_txt.append(input_txt)

    x_1 = concatenate(temp)
#     x_1 = Dense(20, activation="relu")(x_1)
    x_1 = Dense(50, activation="relu")(x_1)
    x_1 = Dropout(Drop)(x_1)

    input_cat = Input((X_cat.shape[1], ))
    x_2 = Embedding(2, 10, input_length=X_cat.shape[1])(input_cat)
    x_2 = SpatialDropout1D(Drop)(x_2)
    x_2 = Flatten()(x_2)

    input_num = Input((len(numFeatures), ))
    x_3 = Dropout(Drop)(input_num)
    
    x = concatenate([x_1, x_2, x_3])

    for HL in HLs:
        x = Dense(HL, activation="relu")(x)
        x = Dropout(Drop)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs_txt+[input_cat, input_num], outputs=output)
    model.compile(
            optimizer=OP,
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])
    return model

def trainNN(X_train, X_val, Tar_train, Tar_val, HL=[50], Drop=0.5, OP=optimizers.Adam()):
    file_path='NN.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=2,
                                   verbose=1,
                                   epsilon=3e-4,
                                   mode='min')

    model = getModel(HL, Drop, OP)
    model.fit(breakInput(X_train), Tar_train, validation_data=(breakInput(X_val), Tar_val),
                        verbose=2, epochs=50, batch_size=1000, callbacks=[early, lr_reduced, checkpoint])
    model.load_weights(file_path)
    return model

params_xgb = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.25,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1,
    }
params_lgb = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.25,
        'bagging_fraction': 0.85,
        'seed': 0,
        'verbose': 0,
    }
nCV = 1 # should be ideally larger
for i in range(21, 22):
    gc.collect()
    X_train, X_val, Tar_train, Tar_val = train_test_split(Xtr, Ttr_tar, test_size=0.15, random_state=i, stratify=Ttr_tar)
    # XGB
    dtrain = xgb.DMatrix(X_train, label=Tar_train)
    dval   = xgb.DMatrix(X_val, label=Tar_val)
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(params_xgb, dtrain, 5000,  watchlist, maximize=True, verbose_eval=200, early_stopping_rounds=200)
    Yvl1 = model.predict(dval)
    Yts1 = model.predict(xgb.DMatrix(Xts))
    # LGB
    dtrain = lgb.Dataset(X_train, Tar_train)
    dval   = lgb.Dataset(X_val, Tar_val)
    model = lgb.train(params_lgb, dtrain, num_boost_round=10000, valid_sets=[dtrain, dval], early_stopping_rounds=200, verbose_eval=200)
    Yvl2 = model.predict(X_val)
    Yts2 = model.predict(Xts)
    # NN
    model = trainNN(X_train, X_val, Tar_train, Tar_val, HL=[50], Drop=0.5, OP=optimizers.Adam())
    Yvl3 = model.predict(breakInput(X_val)).squeeze()
    Yts3 = model.predict(breakInput(Xts)).squeeze()
    # stack
    M = LinearRegression()
    M.fit(pl.array([Yvl1, Yvl2, Yvl3]).T, Tar_val)
    Yts.append(M.predict(pl.array([Yts1, Yts2, Yts3]).T))

from sklearn.preprocessing import MinMaxScaler
Tts[target] = MinMaxScaler().fit_transform(pl.array(Yts).mean(0).reshape(-1,1))
Tts[['id', target]].to_csv('text_cat_num_xgb_lgb_NN.csv', index=False)