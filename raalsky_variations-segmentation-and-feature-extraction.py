# Standard imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
import sklearn
from tqdm import tqdm, tqdm_notebook
import xgboost as xgb
import re

%matplotlib inline
train_variants = pd.read_csv('../input/training_variants')
print('Number of training variants: %d' % (len(train_variants)))
train_variants.head()
test_variants = pd.read_csv('../input/test_variants')
print('Number of test variants: %d' % (len(test_variants)))
test_variants.head()
def read_textfile(filename):
    return pd.read_csv(filename, sep='\|\|', header=None, names=['ID', 'Text'], skiprows=1, engine='python')
train_text = read_textfile('../input/training_text')
print('Number of train samples: %d' % (len(train_text)))
train_text.head()
test_text = read_textfile('../input/test_text')
print('Number of test samples: %d' % (len(test_text)))
test_text.head()
train_data = pd.merge(train_text, train_variants, how='left', on='ID')
test_data = pd.merge(test_text, test_variants, how='left', on='ID')
filters = [
    '^([A-Z])([0-9]*)_([A-Z])([0-9]*)([a-z]+)([A-Z]+)$',
    '^([A-Z])([0-9]+)_([A-Z])([0-9]+)([a-z]+)$',
    '^([A-Z])([0-9]+)([a-z_]+)$',
    '^([A-Z]|null)([0-9]*)([A-Z\*])$',
    '^([A-Z0-9]+)\-([A-Z0-9]+) Fusion$',
    '^Deletion|Truncating Mutations|Promoter Mutations|Amplification|Promoter Hypermethylation|Overexpression|Fusions|Epigenetic Silencing|Wildtype|Copy Number Loss|Hypermethylation|Single Nucleotide Polymorphism$',
]

names = [
    'Gene_%d' % i for i in range(1, 10)
]

vals = {
    'del'     : 0,
    'dup'     : 1,
    'ins'     : 2,
    'fs'      : 3,
    '_splice' : 4
}
def prepareData(data, test=False):
    segments = [
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'ACID_1', 'Pos_1', 'ACID_2', 'Pos_2', 'Operation', 'Sequence', 'Class']),
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'ACID_1', 'Pos_1', 'ACID_2', 'Pos_2', 'Operation', 'Class']),
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'ACID', 'Pos', 'Operation', 'Class']),
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'ACID_1', 'Pos', 'ACID_2', 'Class']),
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'Gene_1', 'Gene_2', 'Class']),
        pd.DataFrame(data=[], columns=['ID', 'Text', 'Gene', 'GeneralOperation', 'Class'])
        ]
    
    for it, row in tqdm_notebook(data.iterrows(), total=len(data)):
        for i, f in enumerate(filters):
            if re.match(f, row['Variation']):
                CLASS = 0
                if not test:
                    CLASS = row['Class']
                if i != 5:
                    if re.search(f, row['Variation']) is None:
                        print(row['Variation'], f)
                    else: #*(' ' * 9),  
                        segments[i].loc[len(segments[i])] = [ row['ID'], row['Text'], row['Gene'], *re.search(f, row['Variation']).groups(), CLASS ]
                else:
                    
                    segments[i].loc[len(segments[i])] = [ row['ID'], row['Text'], row['Gene'], row['Variation'], CLASS ]
    if test:
        for i, seg in enumerate(segments):
            segments[i] = seg.drop(['Class'], axis=1)
    return segments
def formatData(df, shuffle=False):
    #df.Gene = df.Gene.apply(lambda x: x.ljust(9))
    #for i in range(1, 10):
    #    df['Gene_%d' % i] = df.Gene.apply(lambda x : x [i-1])
    for i in df.columns:
        if i == 'Pos_1' or i == 'Pos_2' or i == 'Class' or i == 'ID' or i == 'Pos':
            df[i] = df[i].apply(lambda x : int(x))
        #if i == 'ACID_1' or i == 'ACID_2' or i == 'ACID':
        #    df[i] = df[i].apply(lambda x : ord(x)-32)
        #if i == 'Operation':
        #    df[i] = df[i].apply(lambda x : vals[x])
        #if i in names:
        #    df[i] = df[i].apply(lambda x : ord(x)-32)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df#.drop('Gene', axis=1)
train_raw_segments = prepareData(train_data)
train_segments = [ formatData(x, shuffle=True) for x in train_raw_segments ]
test_raw_segments = prepareData(test_data, test=True)
test_segments = [ formatData(x) for x in test_raw_segments ]
acids = set(
    list(test_segments[0].ACID_1) +
    list(test_segments[0].ACID_2) +
    list(train_segments[0].ACID_1) +
    list(train_segments[0].ACID_2) +
    
    list(test_segments[1].ACID_1) +
    list(test_segments[1].ACID_2) +
    list(train_segments[1].ACID_1) +
    list(train_segments[1].ACID_2) +
    
    list(test_segments[2].ACID) +
    list(train_segments[2].ACID) +
    
    list(test_segments[3].ACID_1) +
    list(test_segments[3].ACID_2) +
    list(train_segments[3].ACID_1) +
    list(train_segments[3].ACID_2)
    )
genes = set(
    list(test_segments[0].Gene) +
    list(test_segments[1].Gene) +
    list(test_segments[2].Gene) +
    list(test_segments[3].Gene) +
    list(test_segments[4].Gene) +
    list(test_segments[5].Gene) +
    list(test_segments[4].Gene_1) +
    list(test_segments[4].Gene_2) +
    
    list(train_segments[0].Gene) +
    list(train_segments[1].Gene) +
    list(train_segments[2].Gene) +
    list(train_segments[3].Gene) +
    list(train_segments[4].Gene) +
    list(train_segments[5].Gene) +
    list(train_segments[4].Gene_1) +
    list(train_segments[4].Gene_2)
    )
ops = set(
        list(train_segments[0].Operation) +
        list(train_segments[1].Operation) +
        list(train_segments[2].Operation) +
    
        list(test_segments[0].Operation) +
        list(test_segments[1].Operation) +
        list(test_segments[2].Operation)
    )
gops = set(
        list(train_segments[5].GeneralOperation) +
        list(test_segments[5].GeneralOperation)
    )
global_acid_encoder = preprocessing.LabelEncoder();
global_acid_encoder.fit(list(acids))
global_op_encoder = preprocessing.LabelEncoder();
global_op_encoder.fit(list(ops))
global_gop_encoder = preprocessing.LabelEncoder();
global_gop_encoder.fit(list(gops))
class truncateFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self;
    def transform(self, x):
        x = x.drop([
                'ID',
                'Class',
                'Text',
                'Gene'
            ], axis=1).values
        return x
class filterColumn(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x[self.column].values
        return np.resize(x, (x.shape[0], 1))
class filterStrColumn(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.column].apply(str)
class printFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x
class labelEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, label):
        self.label = label
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = eval("global_%s_encoder.transform(np.ravel(x))" % (self.label))
        return np.resize(x, (x.shape[0], 1))
fp = {
    0 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Genes', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Text', pipeline.Pipeline([
                                    ('Text', filterStrColumn('Text')),
                                    ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
                                    ('tsvd3', decomposition.TruncatedSVD(n_components=100, n_iter=25, random_state=12))
                        ])),
                        ('ACID_1', pipeline.Pipeline([
                                    ('ACID_1', filterColumn('ACID_1')),
                                    ('LB_ACID_1', labelEncoder('acid'))
                        ])),
                        ('Pos_1', filterColumn('Pos_1')),
                        ('ACID_2', pipeline.Pipeline([
                                    ('ACID_2', filterColumn('ACID_2')),
                                    ('LB_ACID_2', labelEncoder('acid'))
                        ])),
                        ('Pos_2', filterColumn('Pos_2')),
                        ('Operation', pipeline.Pipeline([
                                    ('Operation', filterColumn('Operation')),
                                    ('LB_Operation', labelEncoder('op'))
                        ])),
                        ('Sequence', pipeline.Pipeline([
                                    ('Sequence', filterStrColumn('Sequence')),
                                    ('Count_Sequence', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Sequence', decomposition.TruncatedSVD(n_components=10, n_iter=25, random_state=12))
                        ]))
                    ]
                ))
        ]),
    1 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Genes', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Text', pipeline.Pipeline([
                                    ('Text', filterStrColumn('Text')),
                                    ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
                                    ('tsvd3', decomposition.TruncatedSVD(n_components=100, n_iter=25, random_state=12))
                        ])),
                        ('ACID_1', pipeline.Pipeline([
                                    ('ACID_1', filterColumn('ACID_1')),
                                    ('LB_ACID_1', labelEncoder('acid'))
                        ])),
                        ('Pos_1', filterColumn('Pos_1')),
                        ('ACID_2', pipeline.Pipeline([
                                    ('ACID_2', filterColumn('ACID_2')),
                                    ('LB_ACID_2', labelEncoder('acid'))
                        ])),
                        ('Pos_2', filterColumn('Pos_2')),
                        ('Operation', pipeline.Pipeline([
                                    ('Operation', filterColumn('Operation')),
                                    ('LB_Operation', labelEncoder('op'))
                        ]))
                    ]
                ))
        ]),
    2 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Genes', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Text', pipeline.Pipeline([
                                    ('Text', filterStrColumn('Text')),
                                    ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
                                    ('tsvd3', decomposition.TruncatedSVD(n_components=100, n_iter=25, random_state=12))
                        ])),
                        ('ACID', pipeline.Pipeline([
                                    ('ACID', filterColumn('ACID')),
                                    ('LB_ACID', labelEncoder('acid'))
                        ])),
                        ('Pos', filterColumn('Pos')),
                        ('Operation', pipeline.Pipeline([
                                    ('Operation', filterColumn('Operation')),
                                    ('LB_Operation', labelEncoder('op'))
                        ]))
                    ]
                ))
        ]),
    3 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Genes', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Text', pipeline.Pipeline([
                                    ('Text', filterStrColumn('Text')),
                                    ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
                                    ('tsvd3', decomposition.TruncatedSVD(n_components=100, n_iter=25, random_state=12))
                        ])),
                        ('ACID_1', pipeline.Pipeline([
                                    ('ACID_1', filterColumn('ACID_1')),
                                    ('LB_ACID_1', labelEncoder('acid'))
                        ])),
                        ('Pos', filterColumn('Pos')),
                        ('ACID_2', pipeline.Pipeline([
                                    ('ACID_2', filterColumn('ACID_2')),
                                    ('LB_ACID_2', labelEncoder('acid'))
                        ]))
                    ]
                ))
        ]),
    4 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Gene', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Gene_1', pipeline.Pipeline([
                                    ('Gene_1', filterStrColumn('Gene_1')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Gene_2', pipeline.Pipeline([
                                    ('Gene_2', filterStrColumn('Gene_2')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                    ]
                ))
        ]),
    5 : pipeline.Pipeline([
            ('union', pipeline.FeatureUnion(
                    n_jobs = 4,
                    transformer_list = [
                        ('Genes', pipeline.Pipeline([
                                    ('Gene', filterStrColumn('Gene')),
                                    ('Count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))),
                                    ('LDA_Count_Gene', decomposition.TruncatedSVD(n_components=60, n_iter=25, random_state=12))
                        ])),
                        ('Text', pipeline.Pipeline([
                                    ('Text', filterStrColumn('Text')),
                                    ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))),
                                    ('tsvd3', decomposition.TruncatedSVD(n_components=100, n_iter=25, random_state=12))
                        ])),
                        ('GeneralOperation', pipeline.Pipeline([
                                    ('GeneralOperation', filterColumn('GeneralOperation')),
                                    ('LB_GeneralOperation', labelEncoder('gop'))
                        ]))
                    ]
                ))
        ])
}
train_x = [[],[],[],[],[],[]]
validate_x = [[],[],[],[],[],[]]
for i, seg in enumerate(train_segments):
    train_x[i] = fp[i].fit_transform(seg);
    print('Segment %d' % i, train_x[i].shape)
test_x = [[],[],[],[],[],[]]
for i, seg in enumerate(test_segments):
    test_x[i] = fp[i].transform(seg);
    print('Segment %d' % i, test_x[i].shape)
general_params = {
    'eta': 0.04,
    'max_depth': 6,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 9,
    'seed': 2510,
    'silent': True,
    'n_jobs' : 4,
}
folds = 5
params = {
    0 : {},
    1 : {},
    2 : {},
    3 : {},
    4 : {},
    5 : {}
}
model = [[],[],[],[],[],[]]
for seg_id in range(len(train_segments)):
    print("Training Segment %d" % seg_id)
    print('-' * 100)
    train_y = train_segments[seg_id].Class.values-1
    validate_y = validate_segments[seg_id].Class.values-1
    model[seg_id] = [[], [], [], [], []]
    for i in range(fold):
        x1, x2, y1, y2 = model_selection.train_test_split(train_x[seg_id], train_y, test_size=0.1, random_state=i)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model[seg_id][i] = xgb.train(general_params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
test = pd.DataFrame(data=[], columns=['ID', 'class1','class2','class3','class4','class5','class6','class7','class8', 'class9'])
for seg_id in range(0, len(train_segments)):
    print("Segment %d" % seg_id)
    print('-' * 100)
    for i in range(fold):
        if i != 0:
            test_pred += model[seg_id][i].predict(xgb.DMatrix(test_x[seg_id]), ntree_limit=model[seg_id][i].best_ntree_limit) / fold
        else:
            test_pred = model[seg_id][i].predict(xgb.DMatrix(test_x[seg_id]), ntree_limit=model[seg_id][i].best_ntree_limit) / fold
    test_out = pd.DataFrame(data=np.concatenate((np.resize(test_segments[seg_id].ID.values, (len(test_segments[seg_id]), 1)), test_pred), axis=1), columns=['ID', 'class1','class2','class3','class4','class5','class6','class7','class8', 'class9'])
    test = pd.concat([test, test_out])
test.ID = test.ID.apply(int)
test = test.reset_index(drop=True)
set(test_variants.ID) - set(test.ID)
test.loc[len(test)] = [1628, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
test.loc[len(test)] = [4501, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
test.loc[len(test)] = [4631, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
test.loc[len(test)] = [4984, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
test.ID = test.ID.apply(int)
test.to_csv('submission.csv', index=False)