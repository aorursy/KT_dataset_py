import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.utils import shuffle

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier

from sklearn.svm import SVC



import keras.optimizers

from keras.models import Sequential 

from keras.layers import Dense, GaussianNoise

from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier



%matplotlib inline

sns.set()



pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('display.max_columns', 25)



np.set_printoptions(suppress=True)
train = pd.read_csv('../input/treino.csv')

test = pd.read_csv('../input/teste.csv')

info = pd.read_csv('../input/info_uso.csv')

machines = pd.read_csv('../input/maquinas.csv')

errors = pd.read_csv('../input/erros.csv')
info = info.merge(machines, on='maquina')

info = info.merge(pd.get_dummies(errors, prefix=[''], prefix_sep='', columns=['erro']).groupby(['data', 'maquina'], as_index=False).sum(), how='left', on=['data', 'maquina'])

info.fillna(0, inplace=True)

info.data = pd.to_datetime(info.data, infer_datetime_format=True)
def query_range(maq, a, b):

    aux = info.query("maquina==@maq and @a<=data<@b")

    med = aux[['voltagem', 'rotacao', 'pressao', 'vibracao']].median()

    tot = aux[['error1', 'error2', 'error3', 'error4', 'error5']].sum()

    return pd.Series([

        med['voltagem'],

        med['rotacao'],

        med['pressao'],

        med['vibracao'],

        tot['error1'],

        tot['error2'],

        tot['error3'],

        tot['error4'],

        tot['error5']

    ])



def process(row):

    maq = row['maquina']

    a = row['data'] - pd.DateOffset(days=6)

    b = row['data'] + pd.DateOffset(days=1)

    return pd.concat([

        query_range(maq, row['data'] - pd.DateOffset(days=6), row['data'] + pd.DateOffset(days=1)),

        query_range(maq, row['data'] - pd.DateOffset(days=1), row['data'] + pd.DateOffset(days=1))

    ], ignore_index=True)



def prepare(df):

    df = df.merge(machines, on='maquina')

    df.data = pd.to_datetime(df.data, infer_datetime_format=True)

    df[['voltagem_w', 'rotacao_w', 'pressao_w', 'vibracao_w', 'error1_w', 'error2_w', 'error3_w', 'error4_w', 'error5_w',

       'voltagem_d', 'rotacao_d', 'pressao_d', 'vibracao_d', 'error1_d', 'error2_d', 'error3_d', 'error4_d', 'error5_d']] = df.apply(process, axis=1)

    return df



train = prepare(train)

test = prepare(test)
info.head(10)
train.head(10)
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'idade')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'voltagem_d')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'rotacao_d')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'pressao_d')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'vibracao_d')

fg.add_legend()
plt.figure(figsize=(15,8))

sns.barplot(data=train.loc[:, ['falha', 'error1_d', 'error2_d', 'error3_d', 'error4_d', 'error5_d']].melt(id_vars='falha'), x='falha', y='value', hue='variable', order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'])
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'voltagem_w')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'rotacao_w')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'pressao_w')

fg.add_legend()
fg = sns.FacetGrid(train, hue='falha', col='modelo', col_wrap=2, height=6.25, aspect=1, hue_order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'], col_order=['model1', 'model2', 'model3', 'model4'])

fg.map(sns.distplot, 'vibracao_w')

fg.add_legend()
plt.figure(figsize=(15,8))

sns.barplot(data=train.loc[:, ['falha', 'error1_w', 'error2_w', 'error3_w', 'error4_w', 'error5_w']].melt(id_vars='falha'), x='falha', y='value', hue='variable', order=['ok', 'comp1', 'comp2', 'comp3', 'comp4'])
def score_model(y_test, y_pred, ok=4):

    miss = 0

    total = 0

    near_miss = 0

    miss_failure = 0

    total_failure = 0

    aux = y_test.values

    

    for i in range(len(y_pred)):

        if aux[i] != ok:

            if y_pred[i] != aux[i]:

                if y_pred[i] != ok:

                    near_miss += 1

                miss_failure += 1

            total_failure += 1



        if y_pred[i] != aux[i]:

            miss += 1

        total += 1



    print('f1 score: {0:.5f}'.format(f1_score(y_test, y_pred, average='weighted')))

    print('precision: {0:.5f}'.format(precision_score(y_test, y_pred, average='weighted')))

    print('recall: {0:.5f}'.format(recall_score(y_test, y_pred, average='weighted')))

    print('total missed', miss, 'of', total)

    print('failure missed', miss_failure, 'of', total_failure, 'with', near_miss, 'near misses')
def hit_score(y_test, y_pred, ok=4):

    miss = 0

    total = 0

    miss_failure = 0

    total_failure = 0

    aux = y_test.values

    

    for i in range(len(y_pred)):

        # 4 = 'ok'

        if aux[i] != ok:

            if y_pred[i] != aux[i]:

                miss_failure += 1

            total_failure += 1



        if y_pred[i] != aux[i]:

            miss += 1

        total += 1

        

    r1 = 1 - miss / total

    r2 = 1 - miss_failure / total_failure

    return 2 * r1 * r2 / (r1 + r2)
def prep_undersampling(df, ok=4):

    df_ok = df[df.falha==ok]

    df_fail = df[df.falha!=ok]

    cnt = df_fail.shape[0]

    return df_fail.append(df_ok.sample(cnt))
def prep_oversampling(df, df_y, ok=4):

    df_ok = df[df_y==ok]

    df_fail = df[df_y!=ok]

    df_y_ok = df_y[df_y==ok]

    df_y_fail = df_y[df_y!=ok]

    cnt = df_ok.shape[0] // df_fail.shape[0]

    return df_ok.append([df_fail] * cnt, ignore_index=True), df_y_ok.append([df_y_fail] * cnt, ignore_index=True)
f1 = make_scorer(f1_score, average='micro')

hs = make_scorer(hit_score, greater_is_better=True)
def prep_train(df, keepIndex=False):

    if keepIndex == False:

        df = df.drop(columns=['index'])

    df = df.drop(columns=['data', 'maquina'])

    

    day = ['voltagem_d', 'rotacao_d', 'pressao_d', 'vibracao_d', 'error1_d', 'error2_d', 'error3_d', 'error4_d', 'error5_d']

    week = ['voltagem_w', 'rotacao_w', 'pressao_w', 'vibracao_w', 'error1_w', 'error2_w', 'error3_w', 'error4_w', 'error5_w']

    to_drop = ['modelo']

    to_scale = ['idade'] + day + week

    to_encode = []

    

    df = df.drop(columns=to_drop)

        

    for col in to_scale:

        sc = StandardScaler()

        #sc = MinMaxScaler()

        sc.fit(df.loc[:, [col]])

        df[col] = sc.transform(df.loc[:, [col]])

        

    return pd.get_dummies(df, columns=to_encode)
df = prep_train(train.copy(deep=True))



encode = ['falha']

for col in encode:

    le = LabelEncoder()

    le.fit(df[col])

    df[col] = le.transform(df[col])



df_y = df['falha']

df.drop(columns=['falha'], inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df, df_y, stratify=df_y, random_state=None, test_size=0.5)



x_train, y_train = prep_oversampling(x_train, y_train, 4)

x_train, y_train = shuffle(x_train, y_train)
df.head()
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)

pd.DataFrame(data=(model.feature_importances_ * 100).round(5).reshape(1, len(df.columns)), columns=df.columns)
model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)

pd.DataFrame(data=(model.feature_importances_ * 100).round(5).reshape(1, len(df.columns)), columns=df.columns)
model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=5, class_weight='balanced')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)

pd.DataFrame(data=(model.feature_importances_ * 100).round(5).reshape(1, len(df.columns)), columns=df.columns)
model = ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)

pd.DataFrame(data=(model.feature_importances_ * 100).round(5).reshape(1, len(df.columns)), columns=df.columns)
model = ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)

pd.DataFrame(data=(model.feature_importances_ * 100).round(5).reshape(1, len(df.columns)), columns=df.columns)
def create_model(features_count=1):

    model = Sequential()

    model.add(GaussianNoise(0.05, input_shape=(features_count,)))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(24, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=.001), metrics=['categorical_accuracy'])

    return model
model = KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=50, batch_size=250, verbose=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = VotingClassifier(estimators=[

    ('gb', GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5)),

    ('rf1', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')),

    ('rf2', RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('et1', ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced')),

    ('et2', ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=9, class_weight='balanced'))

], voting='soft', weights=[1.25, 1, 1, 1, 1])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = VotingClassifier(estimators=[

    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1)),

    ('rf1', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')),

    ('rf2', RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('et1', ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced')),

    ('et2', ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=5, class_weight='balanced'))

], voting='soft', weights=[1.25, 1.25, 0.75, 0.75, 1.25])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = VotingClassifier(estimators=[

    ('gb', GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5)),

    ('rf1', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')),

    ('rf2', RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('et1', ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced')),

    ('et2', ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('nn', KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))

], voting='soft', weights=[1.25, 1, 1, 1, 1, 2])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = VotingClassifier(estimators=[

    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1)),

    ('rf1', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')),

    ('rf2', RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('et1', ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced')),

    ('et2', ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=5, class_weight='balanced')),

    ('nn', KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))

], voting='soft', weights=[1.25, 1.25, 0.75, 0.75, 1.25, 1])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = VotingClassifier(estimators=[

    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1)),

    ('rf1', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced')),

    ('rf2', RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced')),

    ('et1', ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced')),

    ('et2', ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=5, class_weight='balanced')),

    ('nn', KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))

], voting='soft', weights=[1.5, 2, 1, 1, 2, 2])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
class PredictProbaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, clf=None):

        self.clf = clf



    def fit(self, x, y):

        if self.clf is not None:

            self.clf.fit(x, y)

        return self



    def transform(self, x):

        if self.clf is not None:

            return self.clf.predict_proba(x)

        return x



    def fit_transform(self, x, y):

        return self.fit(x, y).transform(x)
model = Pipeline([

    ('stack', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        #('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=8, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=5, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=50, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', tol=1e-4, decision_function_shape='ovr', max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

    ])),

    ('final', BaggingClassifier(base_estimator=SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500), n_estimators=5, max_samples=0.2))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', BaggingClassifier(base_estimator=SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500), n_estimators=3, max_samples=0.34))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('6', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=len(df.columns), epochs=30, batch_size=250, verbose=0))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

    ])),

    ('final', BaggingClassifier(base_estimator=SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500), n_estimators=5, max_samples=0.2))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', BaggingClassifier(base_estimator=SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500), n_estimators=3, max_samples=0.34))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='sigmoid', gamma=2, degree=1, max_iter=1500))

])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score_model(y_test, y_pred)
test_final = prep_train(test.copy(deep=True), True)

train_final = prep_train(train.copy(deep=True))
le = LabelEncoder()

le.fit(train['falha'])

train_final['falha'] = le.transform(train_final['falha'])
x = train_final.drop(columns=['falha'])

y = train_final['falha']

x, y = prep_oversampling(x, y, 4)

x, y = shuffle(x, y)

x_t = test_final.drop(columns=['index'])
model = Pipeline([

    ('stack1', FeatureUnion([

        ('0', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, subsample=0.5))),

        ('1', PredictProbaTransformer(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.))),

        ('2', PredictProbaTransformer(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, class_weight='balanced'))),

        ('3', PredictProbaTransformer(RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=9, class_weight='balanced'))),

        ('4', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='gini', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

        ('5', PredictProbaTransformer(ExtraTreesClassifier(n_estimators=75, criterion='entropy', max_depth=8, class_weight='balanced', min_samples_leaf=0.01))),

    ])),

    ('stack2', FeatureUnion([

        ('0', PredictProbaTransformer(SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=2, max_iter=1500, probability=True))),

        ('1', PredictProbaTransformer(SVC(class_weight='balanced', kernel='sigmoid', gamma='auto', max_iter=1500, probability=True))),

        ('2', PredictProbaTransformer(KerasClassifier(build_fn=create_model, features_count=30, epochs=5, batch_size=250, verbose=0))),

    ])),

    ('final', SVC(class_weight='balanced', kernel='poly', gamma='auto', degree=1, max_iter=1500))

])

model.fit(x, y)

pred = model.predict(x_t)
sub = pd.DataFrame({'index': test_final['index'], 'falha': le.inverse_transform(pred)})

sub.falha = sub.falha.str.replace('comp', 'peca')

sub.to_csv("submission.csv", index=False)