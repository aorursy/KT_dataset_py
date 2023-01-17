# Imports

from functools import partial  # to mock arguments in functions, like f1_score

from os.path import join as pjoin



import cufflinks as cf

import lightgbm as lgb

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from plotly.offline import init_notebook_mode

from sklearn.metrics import f1_score



init_notebook_mode(connected=False)

cf.go_offline()



pd.options.display.max_rows = 200

pd.options.display.max_columns = 200

pd.options.display.max_colwidth = 200

plt.style.use('ggplot')
import plotly.graph_objects as go





# f1-score data grid preparation

steps = 50

precision_grid = np.linspace(0, 1, num=steps)

recall_grid =    np.linspace(0, 1, num=steps)

pp, rr = np.meshgrid(precision_grid, recall_grid, sparse=True)

f1_score_grid = 2*(pp*rr)/(pp + rr)

f1_score_grid[np.isnan(f1_score_grid)] = 0



# visualize it with plot.ly

fig = go.Figure(data=[go.Surface(z=f1_score_grid, x=precision_grid, y=recall_grid)])



fig.update_layout(

    title='F1-Score surface (feel free to rotate/scale it as you wish)', 

#     autosize=True,

    width=640,

    height=640,

    margin=dict(l=65, r=50, b=65, t=90),

    scene_camera_eye=dict(x=2.2, y=0.78, z=0.64),

    scene=dict(

        xaxis_title='PRECISION',

        yaxis_title='RECALL',

        zaxis_title='F1-SCORE'

    )

)



fig.update_traces(

    contours_z=dict(

        show=True, 

        usecolormap=True,

        highlightcolor="limegreen", 

        project_z=True)

)



fig.show()
# prepare correct metric

f1_score_macro = partial(f1_score, average='macro')
DATA_DIR = '/kaggle/input/nlp-getting-started/'

# DATA_DIR = '../data'

train = pd.read_csv(pjoin(DATA_DIR, 'train.csv'))

test = pd.read_csv(pjoin(DATA_DIR, 'test.csv'))



# glue datasets together, for convenience

train['is_train'] = True

test['is_train'] = False

df = pd.concat(

    [train, test], 

    sort=False, ignore_index=True

).set_index('id').sort_index()



print(train.shape, test.shape, df.shape)

df.head()
import re



from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_text = TfidfVectorizer(

    stop_words='english',

    max_df=0.33,

    min_df=5,

    dtype=np.float32,

    max_features=500,

)



# fit transformer

tfidf_text.fit(df['text'])



NON_WORD_PATTERN = r"[^A-Za-z0-9\.\'!\?,\$\s]"



df_tfidfs = pd.DataFrame(

    np.array(tfidf_text.transform(df['text']).todense()),

    columns=[

        f'tfidf__{re.sub(NON_WORD_PATTERN, "", k)}'

        for (k, v) in

        sorted(tfidf_text.vocabulary_.items(), key=lambda item: item[1])

    ],

    index=df.index,

)

print(df_tfidfs.shape)

df_tfidfs.head()
# add keyword

df['keyword_cat_codes'] = df.keyword.fillna('missing').astype('category').cat.codes



df_features = pd.concat(

    [

       df_tfidfs,

       df[['keyword_cat_codes']]

    ],

    axis=1

)



print(df_features.shape)
import lightgbm as lgb

from sklearn.model_selection import train_test_split, StratifiedKFold
def f1_score_lgb(preds, dtrain):

    labels = dtrain.get_label()

    f_score = f1_score_macro(

        np.round(preds),

        labels,

    )

    return 'f1_score', f_score, True





lgb_params = {

    'num_leaves': 63,

    'learning_rate': 0.015,

    'max_depth': -1,

    'subsample': 0.9,

    'colsample_bytree': 0.33,

}



skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)





cv_res = lgb.cv(

    params=lgb_params,

    train_set=lgb.Dataset(

        data=df_features[df.is_train],

        label=df.loc[df.is_train, 'target'],

        categorical_feature=['keyword_cat_codes'],

    ),

    folds=skf,

#     metrics=['auc'],

    feval=f1_score_lgb,

    verbose_eval=50,

    early_stopping_rounds=200,

    #     eval_train_metric=True,

    num_boost_round=1000,

)
# train simple model according to CV's boosting_rounds params

model = lgb.LGBMClassifier(

    **lgb_params, 

    n_esimators=int( len(cv_res['f1_score-mean']) * (skf.n_splits + 1)/skf.n_splits )

)



model.fit(

    X=df_features[df.is_train],

    y=df.loc[df.is_train, 'target'],

    categorical_feature=['keyword_cat_codes'],

)
# check feature importance

lgb.plot_importance(

    model, 

    importance_type='gain', 

    figsize=(10, 10), 

    max_num_features=50

)
# load leaked data

leaked_labels = pd.read_csv(

    pjoin('../input/a-real-disaster-leaked-label', 'submission.csv')

).set_index('id')

df.loc[~df.is_train, 'target'] = leaked_labels

df.loc[df.is_train, 'target'].mean(), df.loc[~df.is_train, 'target'].mean()
# check total f1-score, on 100% test data

y_pred = pd.Series(

    model.predict(df_features[~df.is_train]), 

    index=df[~df.is_train].index

)

f1_total_test = np.round(

    f1_score_macro(df.loc[~df.is_train, 'target'], y_pred), 4

)



f1_total_test
from tqdm.notebook import tqdm



# generate splits

sampler = np.random.RandomState(911)

n_trials = 1000

random_seeds = sorted(set(sampler.randint(0, 10e7, n_trials)))



# make private/public stratified splits

f1_scores = []



for rs in tqdm(random_seeds):

    public_ind, private_ind = train_test_split(

        df[~df.is_train].index.values, 

        test_size=0.7, 

        random_state=rs,

        stratify=df.loc[~df.is_train, 'target'],

    )

    f1_public = f1_score_macro(

        y_true=df.loc[public_ind, 'target'],

        y_pred=y_pred[public_ind]

    )

    f1_private = f1_score_macro(

        y_true=df.loc[private_ind, 'target'],

        y_pred=y_pred[private_ind]

    )

    

    f1_scores.append((rs, f1_public, f1_private))

    

df_f1 = pd.DataFrame(f1_scores, columns=['random_seed', 'f1_public', 'f1_private'])

df_f1['pub_pr_diff'] = df_f1['f1_public'] - df_f1['f1_private']



# clip some extremes

df_f1['pub_pr_diff'] = df_f1['pub_pr_diff'].clip(

    lower=df_f1['pub_pr_diff'].quantile(0.005),

    upper=df_f1['pub_pr_diff'].quantile(0.995),

)



# check sample data

print(df_f1.shape)

df_f1.head()
df_f1.pub_pr_diff.describe()
# let's check BOX plots of PUBLIC / PRIVATE F1-Scores

df_f1[['f1_public', 'f1_private']].iplot(

    kind='box', 

    dimensions=(640, 320),

    title='F1-Score range, public/private LB'

)
# let's see hist of private f1-scores

f1_mean_private = np.round(df_f1['f1_private'].mean(), 4)

df_f1['f1_private'].iplot(

    kind='hist', 

    title=f'F1 mean_private: {f1_mean_private}<br>F1 total test:       {f1_total_test}',

    dimensions=(640, 320)

)
# let's see hist of differences between public/private score

df_f1['pub_pr_diff'].iplot(

    kind='hist', 

    title='Public/Private F1-score diff distribution', 

    dimensions=(640, 320)

)